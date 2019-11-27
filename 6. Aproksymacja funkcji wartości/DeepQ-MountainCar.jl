
using ReinforcementLearningEnvironments
using Flux
import StatsBase.sample

env = MountainCarEnv();

mutable struct Brain
    β::Float64
    batch_size::Int
    memory_size::Int
    memory::Array{Tuple,1}
    net::Chain
    η::Float64
end

function Brain(env; β = 0.99, η = 0.001)
    model = Chain(Dense(length(env.state), 128, relu), 
            Dense(128, 52, relu), 
            Dense(52, length(action_space(env)), identity))
    Brain(β, 64 , 50_000, [], model, η)
end

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

loss(x, y) = Flux.mse(agent.brain.net(x), y)

function replay!(agent::Agent)
    batch_size = min(agent.brain.batch_size, length(agent.brain.memory))
    x = zeros(Float32,length(agent.env.state), batch_size)
    y = zeros(Float32,length(action_space(env)), batch_size)
    for (i,step)  in enumerate(sample(agent.brain.memory, batch_size, replace = false))
        s,a,r,s′,terminal = step
        terminal ? (newQ  = r) : (newQ = r + agent.brain.β * maximum(agent.brain.net(s′).data))
        Q = agent.brain.net(s).data
        Q[a] = newQ
        x[:, i] .= s
        y[:, i] .= Q
    end
    Flux.train!(loss, params(agent.brain.net), [(x, y)], ADAM(agent.brain.η))
end

function remember!(brain::Brain, step::Tuple)
    length(brain.memory) == brain.memory_size && deleteat!(brain.memory,1)
    push!(brain.memory, step)
end

policy(agent::Agent, state::Array{Float64,1}) = argmax(agent.brain.net(state).data)

function step!(agent::Agent, train::Bool)
    s = deepcopy(agent.env.state)
    (rand() < agent.ϵ  && train) ? (a = rand(action_space(agent.env))) : (a = policy(agent, s))
    interact!(agent.env, a)
    obs = observe(agent.env)
    r, s′, terminal = deepcopy(get_reward(obs)), deepcopy(get_state(obs)), deepcopy(get_terminal(obs))
    agent.position = s′[1]
    agent.reward += r
    remember!(agent.brain, (s,a,r,s′,terminal))
    train && replay!(agent)
    terminal 
end

function run!(agent::Agent, episodes::Int; train::Bool = true, plotting::Bool = true, summary::Bool = true)
    ep = 1.0
    success = 0.0
    while ep ≤ episodes
        plotting && (render(env); sleep(0.0001))
        if step!(agent, train) 
            reset!(agent.env)
            agent.position > 0.5 && (success += 1.0)
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
end

agent = Agent(env);

res = run!(agent,1000; plotting = false);






