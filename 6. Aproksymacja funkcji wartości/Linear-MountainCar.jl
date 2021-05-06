using ReinforcementLearningBase, ReinforcementLearningEnvironments
using GR, PyPlot; pygui(true) 
import StatsBase.sample
env = MountainCarEnv();

tilling(S,n,offset) = [(i,j) for i in range(S[1].left + offset,S[1].right + offset, length = n + 1), 
        j in range(S[2].left + offset,S[2].right + offset, length = n + 1)] 

tillings(S,n,m) = [tilling(S,n,i/n) for i = 0:m-1]

between(x,a,b) = a ≤ x < b 

code(s,t::Array{Tuple{Float64,Float64},2}) = [between(s[1],t[i,j][1],t[i+1,j][1]) && between(s[2],t[i,j][2],t[i,j+1][2])
      for j = 1:size(t,2)-1 for i = 1:size(t,1)-1]

code(s,tillings::Array{Array{Tuple{Float64,Float64},2},1}) = 
                vcat([code(s,t) for t in tillings]...)

mutable struct Agent
    env::AbstractEnv
    ϵ::Float64
    ϵ_decay::Float64
    ϵ_min::Float64
    β::Float64
    η::Float64
    tillings::Array{Array{Tuple{Float64,Float64},2},1}
    W::Array{Float64,2}
    position::Float64
    reward::Float64
end

Agent(env, n, m, ϵ = 1.0, ϵ_decay = 0.9975, 
    ϵ_min = 0.005, β = 0.99, η = 0.02) = Agent(env, ϵ, ϵ_decay, ϵ_min, β, η,
                                                tillings(state_space(env), n, m),
                                                rand(length(action_space(env)), n*n*m),
                                                -Inf, 0.0)


policy(agent::Agent, state::Array{Bool,1}) = argmax(agent.W * state)

function step!(agent::Agent, train::Bool)
    s = code(deepcopy(state(agent.env)), agent.tillings)
    (rand() < agent.ϵ  && train) ? (a = rand(agent.env.action_space)) : (a = policy(agent, s))
    agent.env(a)
    r, s′, terminal = deepcopy(reward(agent.env)), deepcopy(state(agent.env)), deepcopy(is_terminated(agent.env))
    agent.position = s′[1]
    s′ = code(s′,agent.tillings)
    agent.reward += r
    if train
        if terminal
            R  = r
        else
            Q_new_state = agent.W * s′
            R = r + agent.β * Q_new_state[policy(agent,s′)]
        end
        Q_hat = agent.W * s
        Q = deepcopy(Q_hat)
        Q[a] = R
        agent.W += agent.η*(Q .- Q_hat) * transpose(s)
    end
    terminal 
end

function run!(agent::Agent, episodes::Int; train::Bool = true, plotting::Bool = true, summary::Bool = true)
    rewards = []
    success_rates = []
    ep = 1.0
    success = 0.0
    while ep ≤ episodes
        plotting && (GR.plot(env); sleep(0.0001))
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


agent = Agent(env,8,4);

rewards, success_rates = run!(agent,5_000; plotting = false);

PyPlot.plot(success_rates)
PyPlot.xlabel("Time")
PyPlot.ylabel("success rate")

X = range(state_space(agent.env)[1].left,state_space(agent.env)[1].right,length = 10)
Y = range(state_space(agent.env)[2].left,state_space(agent.env)[2].right,length = 10)
Z = [maximum(agent.W * code((x,y),agent.tillings)) for x in X, y in Y];

PyPlot.plot_surface(X,Y,-Z)
PyPlot.xlabel("position")
PyPlot.ylabel("velocity")
PyPlot.zlabel("cost-to-go")

rewards,_ = run!(agent,1000; train = false, plotting = false);
