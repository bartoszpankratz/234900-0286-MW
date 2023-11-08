using ReinforcementLearning, ReinforcementLearningEnvironments
using PyPlot

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

mutable struct BlackjackEnv <: AbstractEnv
    dealer_hand::Vector{Int}
    player_hand::Vector{Int}
    done::Bool
    reward::Int
    init::Union{Tuple{Vector{Int}, Vector{Int}}, Nothing}
end

function BlackjackEnv(;init=nothing)
    env = BlackjackEnv([], [], false, 0., init)
    reset!(env)
    env
end

RLBase.state_space(env::BlackjackEnv) = Space([Base.OneTo(31), Base.OneTo(10), Base.OneTo(2)])
RLBase.action_space(env::BlackjackEnv) = Base.OneTo(2)

usable_ace(hand) = (1 in hand) && (sum(hand) + 10 <= 21)
sum_hand(hand) = usable_ace(hand) ? sum(hand) + 10 : sum(hand)
is_bust(hand) = sum_hand(hand) > 21
score(hand) = is_bust(hand) ? 0 : sum_hand(hand)

RLBase.state(env::BlackjackEnv) = (sum_hand(env.player_hand), env.dealer_hand[1], usable_ace(env.player_hand)+1)
RLBase.reward(env::BlackjackEnv) = env.reward
RLBase.is_terminated(env::BlackjackEnv) = env.done

function RLBase.reset!(env::BlackjackEnv)
    empty!(env.dealer_hand)
    empty!(env.player_hand)
    if isnothing(env.init)
        push!(env.dealer_hand, rand(DECK))
        push!(env.dealer_hand, rand(DECK))
        while sum_hand(env.player_hand) < 12
            push!(env.player_hand, rand(DECK))
        end
    else
        append!(env.player_hand, env.init[1])
        append!(env.dealer_hand, env.init[2])
    end
    env.done=false
    env.reward = 0.
end

function (env::BlackjackEnv)(action)
    if action == 1
        push!(env.player_hand, rand(DECK))
        if is_bust(env.player_hand)
            env.done = true
            env.reward = -1
        else
            env.done = false
            env.reward = 0
        end
    elseif action == 2
        env.done = true
        while sum_hand(env.dealer_hand) < 17
            push!(env.dealer_hand, rand(DECK))
        end
        env.reward = cmp(score(env.player_hand), score(env.dealer_hand))
    else
        @error "unknown action"
    end
end

env = BlackjackEnv()

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
    S = state(agent.env)
    if !haskey(agent.Q, S) || rand() < agent.ϵ || es
        A = rand(1:length(action_space(agent.env))) 
    else
        A = argmax(agent.Q[S])
    end
    agent.env(A)
    r = reward(agent.env)
    episode = [(S, A, r)]
    if is_terminated(agent.env) 
        reset!(agent.env)
        return episode
    end
    while step < maxstep
        S = state(agent.env)
        if !haskey(agent.Q, S) || (rand() < agent.ϵ && !es)
            A = rand(1:length(action_space(agent.env))) 
        else
            A = argmax(agent.Q[S])
        end            
        agent.env(A)
        r = reward(agent.env)
        push!(episode, (S, A,r))
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
            S,A,r = episode[i]
            R = agent.β*R + r
        end
        S,A,r = deepcopy(episode[occur_first[k]])
        if !haskey(agent.Q, S)
            agent.C[S] = zeros(length(action_space(agent.env)))
            agent.Q[S] = zeros(length(action_space(agent.env)))
        end
        agent.C[S][A] += 1.0
        agent.Q[S][A] += (R - agent.Q[S][A])/ agent.C[S][A]
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

agent = Agent(env)

success_rate = MC!(agent, maxit = 5_000_000);

plot(success_rate[100:5_000_000])
xlabel("Time")
ylabel("success rate")



cm = get_cmap(:gray)
p = plot_surface(1:10, 12:21, [maximum(agent.Q[(i,j,1)]) 
        for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")
colorbar(p)

imshow([argmax(agent.Q[(i,j,1)]) for i in 21:-1:12, j in 1:10], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")


cm = get_cmap(:gray)
p = plot_surface(1:10, 12:21, [maximum(agent.Q[(i,j,2)]) for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")
colorbar(p)

imshow([argmax(agent.Q[(i,j,2)]) for i in 21:-1:12, j in 1:10], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")


agent2 = Agent(env)

success_rate2 = MC!(agent2, es = true, maxit = 5_000_000);

plot(success_rate2[100:5_000_000])
xlabel("Time")
ylabel("success rate")



p = plot_surface(1:10, 12:21, [maximum(agent2.Q[(i,j,1)]) for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")
colorbar(p)

imshow([argmax(agent2.Q[(i,j,1)]) for i in 21:-1:12, j in 1:10], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("No Usable Ace")

cm = get_cmap(:gray)
p = plot_surface(1:10, 12:21, [maximum(agent2.Q[(i,j,2)]) for i in 12:21, j in 1:10], cmap = get_cmap("rainbow") )
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")
colorbar(p)



imshow([argmax(agent2.Q[(i,j,2)]) for i in 21:-1:12, j in 10:-1:1], cmap = get_cmap("gray_r"))
yticks(0:9,string.(collect(21:-1:12)))
xticks(0:9,string.(collect(1:10)))
ylabel("Player's Hand")
xlabel("Dealers showing")
title("Usable Ace")
