using ReinforcementLearningBase, ReinforcementLearningEnvironments
using Flux
using Plots; gr(); 
import StatsBase.sample, StatsBase.Weights

env = MountainCarEnv();

mutable struct Brain
    β::Float64
    batch_size::Int
    memory_size::Int
    min_memory_size::Int
    memory::Array{Tuple,1}
    policy_net::Chain
    value_net::Chain
    ηₚ::Float64
    ηᵥ::Float64
end

function Brain(env; β = 0.99, ηₚ = 0.00001, ηᵥ = 0.001)
    policy_net = Chain(Dense(length(env.state), 40, identity),
                Dense(40,40,identity),
                Dense(40,length(action_space(env)), identity), softmax)
    value_net = Chain(Dense(length(env.state), 128, relu), 
                    Dense(128, 52, relu), 
                    Dense(52, 1, identity))
    Brain(β, 64 , 50_000, 1000, [], policy_net, value_net, ηₚ, ηᵥ)
end

mutable struct Agent
    env::AbstractEnv
    brain::Brain
    position::Float64
    reward::Float64
end

Agent(env::AbstractEnv) = Agent(env, Brain(env), -Inf, 0.0)

function actor_loss(x, A, γ = 0.001) 
    p = agent.brain.policy_net(x)
    loss = sum(-log.(p .+ 1e-7) .* A)/size(A,1) 
    entropy = sum(-log.(p .+ 1e-7) .* p)/size(A,1)
    return loss - γ * entropy
end      

critic_loss(x, y, ξ = 0.5) = ξ*Flux.mse(agent.brain.value_net(x), y)

function replay!(agent::Agent)
    x = zeros(Float32,length(agent.env.state), agent.brain.batch_size)
    A = zeros(Float32,length(action_space(agent.env)), agent.brain.batch_size)
    y = zeros(Float32,1, agent.brain.batch_size)
    for (i,step)  in enumerate(sample(agent.brain.memory, agent.brain.batch_size, replace = false))
        s,a,r,s′,v,v′,terminal = step
        terminal ? (R  = r) : (R = r + agent.brain.β * v′)
        adv = R - v
        Adv = zeros(Float32,length(action_space(agent.env)))
        Adv[a] = adv
        x[:, i] .= s
        A[:, i] .= Adv
        y[:, i] .= R
    end
    Flux.train!(actor_loss, params(agent.brain.policy_net), [(x, A)], ADAM(agent.brain.ηₚ))
    Flux.train!(critic_loss, params(agent.brain.value_net), [(x, y)], ADAM(agent.brain.ηᵥ))
end

function remember!(brain::Brain, step::Tuple)
    length(brain.memory) == brain.memory_size && deleteat!(brain.memory,1)
    push!(brain.memory, step)
end

function forward(brain::Brain, state)
    π = agent.brain.policy_net(state)
    v = agent.brain.value_net(state)[1]
    return π,v
end

function step!(agent::Agent, train::Bool)
    s = deepcopy(agent.env.state)
    π,v = forward(agent.brain, s)
    a = sample(1:length(action_space(agent.env)),Weights(π))
    agent.env(a)
    r, s′, terminal = deepcopy(reward(agent.env)), deepcopy(state(agent.env)), 
    deepcopy(is_terminated(agent.env))
    _,v′ = forward(agent.brain, s′)
    agent.position = s′[1]
    agent.reward += r
    remember!(agent.brain, (s,a,r,s′,v,v′,terminal))
    (train && length(agent.brain.memory) > agent.brain.min_memory_size) && replay!(agent)
    terminal 
end

function run!(agent::Agent, episodes::Int; train::Bool = true,
            plotting::Bool = true, summary::Bool = true)
    rewards = []
    success_rates = []
    ep = 1.0
    success = 0.0
    while ep ≤ episodes
        plotting && (plot(env); sleep(0.0001))
        if step!(agent, train) 
            reset!(agent.env)
            agent.position > 0.5 && (success += 1.0)
            push!(rewards, agent.reward)
            push!(success_rates, success/ep)
            if summary
                println("episode $(Int(ep)) ends! Reward: $(agent.reward)")
                println("success rate: $(success/ep)")
            end
            ep += 1.0
            agent.reward = 0.0
            agent.position = -Inf
        end
    end
    return rewards, success_rates
end

agent = Agent(env);

rewards, success_rates = run!(agent,1000; train = true, plotting = false);


