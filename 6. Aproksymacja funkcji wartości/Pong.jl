using Plots, Images, ImageCore
using ReinforcementLearning, ArcadeLearningEnvironment, Flux, BSON
import StatsBase.sample

env = AtariEnv("pong",grayscale_obs = true, frame_skip=4);
update_freq = 4

mutable struct Brain
    β::Float64
    batch_size::Int
    update_freq::Int
    memory_size::Int
    min_memory_size::Int
    memory::Array{Tuple,1}
    input::Array{Float32,4}
    net::Chain
    target::Chain
    target_update_freq::Float64
    η::Float64
end

function Brain(env, update_freq; β = 0.99, η = 0.00001)
    model = Chain(x -> x ./ 255,
                    Conv((8,8), update_freq => 32, relu; stride=4),
                    Conv((4,4), 32 => 64, relu; stride=2),
                    Conv((3,3), 64 => 64, relu; stride=1),
                    x -> reshape(x, :, size(x)[end]),
                    Dense(7*7*64, 512, relu),
                    Dense(512, length(env.action_space), identity)
                    )
    Brain(β, 32,update_freq, 20_000, 1000, [], zeros(84,84,4,1), model, model, 1000, η)
end

mutable struct Agent
    env
    ϵ::Float64
    ϵ_decay::Float64
    ϵ_min::Float64
    brain::Brain
    reward::Float64
end

Agent(env,update_freq, ϵ = 1.0, ϵ_decay = 0.9975, ϵ_min = 0.005) = Agent(env, ϵ, ϵ_decay, ϵ_min, 
                                                                        Brain(env, update_freq), 0.0)

loss(x, y) = Flux.huber_loss(agent.brain.net(x), y)

function replay!(agent::Agent)
    x = zeros(Float32,84,84, agent.brain.update_freq, agent.brain.batch_size)
    y = zeros(Float32,length(agent.env.action_space), agent.brain.batch_size)
    for (i,step)  in enumerate(sample(agent.brain.memory, agent.brain.batch_size, replace = false))
        s,a,r,s′,terminal = step
        terminal ? (R  = r) : (R = r + agent.brain.β * maximum(agent.brain.net(s′)))
        Q = agent.brain.net(s)
        Q[a] = R
        x[:,:,:, i] .= s
        y[:, i] .= Q
    end
    Flux.train!(loss, params(agent.brain.net), [(x, y)], ADAM(agent.brain.η))
end

function remember!(brain::Brain, step::Tuple)
    length(brain.memory) == brain.memory_size && deleteat!(brain.memory,1)
    push!(brain.memory, step)
end

function transform_input!(brain::Brain, state)
    brain.input = brain.input[:, :, [2,3,4,1], :]
    brain.input[:, :, 4, :] = Float32.(imresize(state,(84,84))./255)  
end

policy(agent::Agent, state) = argmax(agent.brain.net(state))

function step!(agent::Agent, train::Bool)        
    transform_input!(agent.brain, state(agent.env))
    s = deepcopy(agent.brain.input)
    (rand() < agent.ϵ  && train) ? (a = rand(agent.env.action_space)) : (a = policy(agent, s))
    agent.env(a)
    r, terminal = deepcopy(reward(agent.env)), deepcopy(is_terminated(agent.env))
    if train
        transform_input!(agent.brain, state(agent.env))
        s′ = deepcopy(agent.brain.input)
        remember!(agent.brain, (s,a,r,s′,terminal))
    end
    agent.reward += r
    terminal 
end

function run!(agent::Agent, episodes::Int; train::Bool = true, plotting::Bool = true, summary::Bool = true)
    rewards = []
    episodes_lengths = []
    steps = 0.0
    ep = 1
    ep_start = 1
    anim = @animate  while ep ≤ episodes
        done = step!(agent, train) 
        steps += 1
        mod(steps, agent.brain. target_update_freq) == 0 && (agent.brain.target = deepcopy(agent.brain.net))
        if mod(steps,agent.brain.update_freq) == 0 && train && length(agent.brain.memory) > agent.brain.min_memory_size 
            replay!(agent)
        end
        plotting && plot(transpose(Gray.(state(env)./255)))
        if done
            reset!(agent.env)
            push!(rewards, agent.reward)
            push!(episodes_lengths, steps - ep_start)
            if summary
                @info "episode $(Int(ep)) ends! Reward: $(agent.reward)"
                @info "ϵ: $(agent.ϵ), episode length: $(steps - ep_start)"
            end
            ep_start = steps + 1
            ep += 1
            if mod(ep,50) == 0 
                m = agent.brain.net 
                BSON.@save "pong_brain.bson" m
            end
            agent.reward = 0.0
            eps = agent.ϵ * agent.ϵ_decay
            agent.ϵ = max(agent.ϵ_min, eps)
        end
        end when plotting
    plotting && gif(anim, "pong.gif", fps = 15)
    return rewards, episodes_lengths
end

agent = Agent(env,update_freq);

#_,_ = run!(agent,1; train = false, plotting = true, summary = true)

rewards, success_rates = run!(agent,5_000, train = true, plotting = false, summary = true);

using BSON
m = agent.brain.net 
BSON.@save "pong_brain.bson" m

