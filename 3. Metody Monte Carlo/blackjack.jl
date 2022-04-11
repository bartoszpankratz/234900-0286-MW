using PyCall, ReinforcementLearning, ReinforcementLearningEnvironments
using PyPlot

env = GymEnv("Blackjack-v1");

mutable struct Agent
    env::AbstractEnv
    ϵ::Float64
    β::Float64 #stopa dyskonta
    Q::Dict #macierz wartosci dla kazdej z par:stan,akcja
    C::Dict #macierz wystapien dla kazdej z par:stan,akcja
end

function Agent(env; ϵ = .1, β = 1.0)
    return Agent(env, ϵ, β, Dict(), Dict())
end

function get_episode(agent; es = false, maxstep = 1000)
    step = 1
    hand,dealer,ace = agent.env.state
    state = (hand,dealer,ace)
    if !haskey(agent.Q, state) || rand() < agent.ϵ || es
        action = rand(1:length(agent.env.action_space)) 
    else
        action = argmax(agent.Q[state])
    end
    agent.env(action - 1)
    r = reward(agent.env)
    episode = [(state, action, r)]
    if is_terminated(agent.env) 
        reset!(agent.env)
        return episode
    end
    while step < maxstep
        hand,dealer,ace = agent.env.state[1]
        state = (hand,dealer,ace)
        if !haskey(agent.Q, state) || (rand() < agent.ϵ && es == false) 
            action = rand(1:length(agent.env.action_space)) 
        else
            action = argmax(agent.Q[state])
        end            
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

function update!(agent; es = false)
    episode = get_episode(agent, es = es)
    R = 0
    occur_first = reverse(vcat(findfirst.(isequal.(unique(episode)), [episode]), length(episode) + 1))
    for k = 2:length(occur_first)
        for i = occur_first[k-1] - 1:-1:occur_first[k]
            state,action,r = episode[i]
            R = agent.β*R + r
        end
        state,action,r = deepcopy(episode[occur_first[k]])
        if !haskey(agent.Q, state)
            agent.C[state] = zeros(length(agent.env.action_space))
            agent.Q[state] = zeros(length(agent.env.action_space))
        end
        agent.C[state][action] += 1.0
        agent.Q[state][action] += (R - agent.Q[state][action])/ agent.C[state][action]
    end  
    episode[end][3] == 1.0 ? (return 1.0) : (return 0.0)
end

function MC!(agent; es = false, maxit = 500000)
    iter = 0
    successes = 0.0
    success_rate = []
    while iter < maxit
        successes += update!(agent, es = es)
        push!(success_rate, successes/iter)
        iter +=1
    end
    return success_rate
end

#without exploring starts

agent = Agent(env)

success_rate = MC!(agent, maxit = 1_000_000);

plot(success_rate)
xlabel("Time")
ylabel("success rate")

cm = get_cmap(:gray)
p = plot_surface(1:10, 12:21, [maximum(agent.Q[(i,j,false)]) for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")
colorbar(p)

imshow([argmax(agent.Q[(i,j,false)]) for i in 21:-1:12, j in 1:10], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")

cm = get_cmap(:gray)
p = plot_surface(1:10, 12:21, [maximum(agent.Q[(i,j,true)]) for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")
colorbar(p)

imshow([argmax(agent.Q[(i,j,true)]) for i in 21:-1:12, j in 1:10], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")

#exploring starts
agent2 = Agent(env)

success_rate2 = MC!(agent2, es = true, maxit = 1_000_000);

plot(success_rate2)
xlabel("Time")
ylabel("success rate")

p = plot_surface(1:10, 12:21, [maximum(agent2.Q[(i,j,false)]) for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")
colorbar(p)

imshow([argmax(agent2.Q[(i,j,false)]) for i in 21:-1:12, j in 1:10], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")

cm = get_cmap(:gray)
p = plot_surface(1:10, 12:21, [maximum(agent2.Q[(i,j,true)]) for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")
colorbar(p)

imshow([argmax(agent2.Q[(i,j,true)]) for i in 21:-1:12, j in 10:-1:1], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")
