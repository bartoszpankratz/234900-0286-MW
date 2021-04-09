using PyCall, ReinforcementLearningBase, ReinforcementLearningEnvironments
using PyPlot

env = GymEnv("FrozenLake-v0");

mutable struct Agent
    env::AbstractEnv
    ϵ::Float64
    β::Float64 #stopa dyskonta
    Q::Array{Float64,2}#macierz wartosci dla kazdej z par:stan,akcja
    C::Array{Float64,2} #macierz wystapien dla kazdej z par:stan,akcja
    π::Array{Int} #strategia agenta
end

function Agent(env; ϵ = .1, β = 0.999)
    return Agent(env,ϵ, β,
        zeros(length(env.observation_space), length(env.action_space)), 
        zeros(length(env.observation_space), length(env.action_space)),
        rand(1:length(env.action_space),length(env.observation_space)))
end

function get_episode(agent, π = agent.π; maxstep = 1000)
    step = 1
    state = 1
    rand() < agent.ϵ ? (action = rand(1:length(agent.env.action_space))) : (action = π[state])
    agent.env(action - 1)
    r = reward(agent.env)
    episode = [(state, action, r)]
    if is_terminated(agent.env) 
        reset!(agent.env)
        return episode
    end
    while step < maxstep
        state = agent.env.state[1] + 1 
        rand() < agent.ϵ ? (action = rand(1:length(agent.env.action_space))) : (action = π[state])
        agent.env(action - 1)
        r = reward(agent.env)
        push!(episode, (state, action,r))
        if is_terminated(agent.env)
            reset!(agent.env)
            break
        end
        step +=1
    end
    return episode
end

function update!(agent)
    episode = get_episode(agent)
    R = 0
    occur_first = reverse(vcat(findfirst.(isequal.(unique(episode)), [episode]), length(episode) + 1))
    for k = 2:length(occur_first)
        for i = occur_first[k-1] - 1:-1:occur_first[k]
            state,action,r = episode[i]
            R = agent.β*R + r
        end
        state,action,r = episode[occur_first[k]]
        agent.C[state,action] += 1
        agent.Q[state,action] += (R - agent.Q[state,action])/ agent.C[state,action]
        agent.π[state] = argmax(agent.Q[state,:])
    end  
    episode[end][3] == 1.0 ? (return 1.0) : (return 0.0)
end

function MC!(agent; maxit = 100000)
    iter = 0
    successes = 0.0
    success_rate = []
    while iter < maxit
        successes += update!(agent)
        push!(success_rate, successes/iter)
        iter +=1
    end
    return success_rate
end

agent = Agent(env);

success_rate = MC!(agent; maxit = 500_000)

plot(success_rate)
xlabel("Time")
ylabel("success rate")