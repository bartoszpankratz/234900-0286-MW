using ReinforcementLearningBase, GridWorlds
using Makie, PyPlot

env = GridWorlds.GridRoomsDirected();

mutable struct Agent
    env::AbstractEnv
    ϵ::Float64 #współczynnik eksploracji
    ϵ_decay::Float64
    ϵ_min::Float64
    β::Float64 #dyskonto
    α::Float64 #stopa uczenia się
    planning::Int64 #liczba krokow planowania
    Q::Dict
    model::Dict
    score::Int #wynik - ile razy agent dotarl do mety
end

function Agent(env, planning_steps; ϵ = 1.0, ϵ_decay = 0.9975, ϵ_min = 0.005,
        β = 0.99, α = 0.1) 
    Agent(env,
        ϵ, ϵ_decay, ϵ_min, β, α, 
        planning_steps, 
        Dict(), Dict(), 0.0)
end

feed!(agent,state, action, reward, new_state) = (agent.model[(state,action)] = (new_state, reward))

function learn!(agent, state, action, reward, new_state)
    if !haskey(agent.Q, state)
        agent.Q[state] = zeros(length(action_space(agent.env)))
        agent.Q[state][action] = reward
    else
        Q_next_state = 0.0
        haskey(agent.Q, new_state) && ( Q_next_state += maximum(agent.Q[new_state]))
        agent.Q[state][action] += agent.α * (reward + agent.β*Q_next_state - agent.Q[state][action])
    end
    
end

function dyna_Q!(agent, steps; burning = true, plotting = false) 
    if plotting
        env_node = Makie.Node(agent.env)
        scene = GridWorlds.init_screen(env_node)
    end
    step = 1.0
    episode = 0.0
    while step <= steps
        if (burning && step < 0.1*steps) || rand() < agent.ϵ || !haskey(agent.Q, state(agent.env))
            action = rand(1:length(action_space(agent.env)))
        else 
            action = argmax(agent.Q[state(agent.env)])
        end
        S = deepcopy(state(agent.env))
        agent.env(action_space(agent.env)[action])
        r = reward(agent.env)
        learn!(agent, S, action, r, deepcopy(state(agent.env)))
        feed!(agent, S, action, r, deepcopy(state(agent.env)))
        for i = 1:agent.planning
            (S,a),(next_S, R) = rand(agent.model)
            learn!(agent, S, a, R, next_S)
        end 
        if plotting 
            Makie.display(scene)
            env_node[] = agent.env
            sleep(1/10)
        end
        if is_terminated(agent.env)
            eps = agent.ϵ * agent.ϵ_decay
            agent.ϵ = max(agent.ϵ_min, eps)
            agent.score += 1.0
            episode += 1.0
            reset!(agent.env)
        end
        step += 1.0 
    end
end

agent = Agent(env,0);

dyna_Q!(agent, 1_000_000)

agent.score

dyna_Q!(agent, 200, burning = false, plotting = true)

agent = Agent(env,10);

dyna_Q!(agent, 1_000_000)

agent.score

agent = Agent(env,30);

dyna_Q!(agent, 1_000_000)

agent.score

#this experiment is pretty slow!

#res = []
#@time for i in 0:3:25
#    scores = []
#    for j in 1:5
#        agent = Agent(env, i);
#        dyna_Q!(agent, 100_000)
#        push!(scores,agent.score)
#    end
#    push!(res,sum(scores)/length(scores))
#end

#PyPlot.plot(res)
#ylabel("Score")
#xlabel("Planning Steps")
