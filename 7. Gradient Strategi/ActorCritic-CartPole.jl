
using ReinforcementLearningEnvironments
using Flux
import StatsBase.sample, StatsBase.Weights

env = CartPoleEnv();

mutable struct Brain
    β::Float64
    batch_size::Int
    memory_size::Int
    memory::Array{Tuple,1}
    policy_net::Chain
    value_net::Chain
    ηₚ::Float64
    ηᵥ::Float64
end

function Brain(env; β = 0.99, ηₚ = 0.001, ηᵥ = 0.005)
    policy_net = Chain(Dense(length(env.state), 24, relu),
                Dense(24,length(action_space(env)), identity), softmax)
    value_net = Chain(Dense(length(env.state), 24, relu), 
                    Dense(24, 1, identity))
    Brain(β, 64 , 50_000, [], policy_net, value_net, ηₚ, ηᵥ)
end

mutable struct Agent
    env::AbstractEnv
    brain::Brain
    reward::Float64
end

Agent(env::AbstractEnv) = Agent(env, Brain(env), 0.0)

actor_loss(x,y) = sum(-log.(agent.brain.policy_net(x) .+ 1e-7) .* y)/size(y,1)

critic_loss(x, y) = Flux.mse(agent.brain.value_net(x), y)

function replay!(agent::Agent)
    batch_size = min(agent.brain.batch_size, length(agent.brain.memory))
    x = zeros(Float32,length(agent.env.state), batch_size)
    y = zeros(Float32,length(action_space(env)), batch_size)
    z = zeros(Float32,1, batch_size)
    for (i,step)  in enumerate(sample(agent.brain.memory, batch_size, replace = false))
        s,a,r,s′,v,v′,terminal = step
        terminal ? (newQ  = r) : (newQ = r + agent.brain.β * v′)
        adv = newQ - v
        Adv = zeros(Float32,length(action_space(env)))
        Adv[a] = adv
        x[:, i] .= s
        y[:, i] .= Adv
        z[:, i] .= newQ
    end
    Flux.train!(actor_loss, params(agent.brain.policy_net), [(x, y)], ADAM(agent.brain.ηₚ))
    Flux.train!(critic_loss, params(agent.brain.value_net), [(x, z)], ADAM(agent.brain.ηᵥ))
    #println(actor_loss(x[:, 1], y[:, 1]))
end

function remember!(brain::Brain, step::Tuple)
    length(brain.memory) == brain.memory_size && deleteat!(brain.memory,1)
    push!(brain.memory, step)
end

function forward(brain::Brain, state)
    π = agent.brain.policy_net(state).data
    v = agent.brain.value_net(state).data[1]
    return π,v
end

function step!(agent::Agent, train::Bool)
    s = deepcopy(agent.env.state)
    π,v = forward(agent.brain, s)
    a = sample(1:length(action_space(agent.env)),Weights(π))
    interact!(agent.env, a)
    obs = observe(agent.env)
    r, s′, terminal = deepcopy(get_reward(obs)), deepcopy(get_state(obs)), deepcopy(get_terminal(obs))
    _,v′ = forward(agent.brain, s′)
    agent.reward += r
    remember!(agent.brain, (s,a,r,s′,v,v′,terminal))
    train && replay!(agent)
    terminal 
end

function run!(agent::Agent, episodes::Int; train::Bool = true,
            plotting::Bool = true, summary::Bool = true)
    ep = 1.0
    success = 0.0
    rewards = []
    while ep ≤ episodes
        plotting && (render(env); sleep(0.0001))
        if step!(agent, train) 
            reset!(agent.env)
            agent.reward == 200 && (success += 1.0)
            push!(rewards, agent.reward)
            if summary
                println("episode $(Int(ep)) ends! Reward: $(agent.reward)")
                println("success rate: $(success/ep)")
            end
            ep += 1.0
            agent.reward = 0.0
        end
    end
    return rewards
end

agent = Agent(env);

res = run!(agent,400; train = true, plotting = false);

res = run!(agent,400; train = false, plotting = false);


