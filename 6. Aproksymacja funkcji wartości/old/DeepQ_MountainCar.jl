using ReinforcementLearningBase, ReinforcementLearningEnvironments
using Flux
using Flux: params
using Plots; gr(); 
import StatsBase.sample

env = MountainCarEnv();

mutable struct Brain
    β::Float64
    batch_size::Int
    memory_size::Int
    min_memory_size::Int
    memory::Array{Tuple,1}
    net::Chain
    η::Float64
end

function Brain(env; β = 0.99, η = 0.001)
    model = Chain(Dense(length(env.state), 128, relu), 
            Dense(128, 52, relu), 
            Dense(52, length(env.action_space), identity))
    Brain(β, 64 , 50_000, 1000, [], model, η)
end

loss(x, y) = Flux.mse(agent.brain.net(x), y)

mutable struct Agent
    env::AbstractEnv
    ϵ::Float64
    ϵ_decay::Float64
    ϵ_min::Float64
    brain::Brain
    position::Float64
    reward::Float64
end

Agent(env::AbstractEnv, ϵ = 1.0, ϵ_decay = 0.9975, ϵ_min = 0.005) = Agent(env, ϵ, ϵ_decay, ϵ_min, 
                                                                        Brain(env), -Inf, 0.0)

function replay!(agent::Agent)
    x = zeros(Float32,length(agent.env.state), agent.brain.batch_size)
    y = zeros(Float32,length(agent.env.action_space), agent.brain.batch_size)
    for (i,step)  in enumerate(sample(agent.brain.memory, agent.brain.batch_size, replace = false))
        s,a,r,s′,terminal = step
        terminal ? (R  = r) : (R = r + agent.brain.β * maximum(agent.brain.net(s′)))
        Q = agent.brain.net(s)
        Q[a] = R
        x[:, i] .= s
        y[:, i] .= Q
    end
    Flux.train!(loss, params(agent.brain.net), [(x, y)], ADAM(agent.brain.η))
end

function remember!(brain::Brain, step::Tuple)
    length(brain.memory) == brain.memory_size && deleteat!(brain.memory,1)
    push!(brain.memory, step)
end

policy(agent::Agent, state::Array{Float64,1}) = argmax(agent.brain.net(state))

function step!(agent::Agent, train::Bool)
    s = deepcopy(agent.env.state)
    (rand() < agent.ϵ  && train) ? (a = rand(agent.env.action_space)) : (a = policy(agent, s))
    agent.env(a)
    r, s′, terminal = deepcopy(reward(agent.env)), deepcopy(state(agent.env)),
    deepcopy(is_terminated(agent.env))
    agent.position = s′[1]
    agent.reward += r
    remember!(agent.brain, (s,a,r,s′,terminal))
    (train && length(agent.brain.memory) > agent.brain.min_memory_size) && replay!(agent)
    terminal 
end

function run!(agent::Agent, episodes::Int; train::Bool = true, plotting::Bool = true, summary::Bool = true)
    rewards = []
    success_rates = []
    ep = 1.0
    success = 0.0
    while ep ≤ episodes
        plotting && (plot(agent.env); sleep(0.0001))
        if step!(agent, train) 
            reset!(agent.env)
            agent.position > 0.5 && (success += 1.0)
            push!(rewards, agent.reward)
            push!(success_rates, success/ep)
            if summary
                println("episode $(Int(ep)) ends! Reward: $(agent.reward)")
                println("ϵ: $(agent.ϵ), success rate: $(success/ep)")
            end
            ep += 1.0
            agent.reward = 0.0
            agent.position = -Inf
            eps = agent.ϵ * agent.ϵ_decay
            agent.ϵ = max(agent.ϵ_min, eps)
        end
    end
    return rewards, success_rates
end

agent = Agent(env);

#rewards,_ = run!(agent,10; train = false, plotting = true);

rewards, success_rates = run!(agent,1000; plotting = false);
#rewards,_ = run!(agent,10; train = false, plotting = true);

plot(success_rates, xlabel = "Time", ylabel = "Sucess rate", legend = false)

plot(rewards, xlabel = "Time", ylabel = "Reward", legend = false)

X = range(state_space(agent.env)[1].left,state_space(agent.env)[1].right,length = 10)
Y = range(state_space(agent.env)[2].left,state_space(agent.env)[2].right,length = 10)
Z = [maximum((agent.brain.net([x,y]))) for x in X, y in Y];

plot(X,Y,-Z, st=:surface, camera = (60,60), xlabel = "Position", ylabel = "Velocity", zlabel = "Cost-to-go")

rewards,success_rates = run!(agent,1000; train = false, plotting = false, summary = false);
plot(success_rates, xlabel = "Time", ylabel = "Sucess rate", legend = false)
