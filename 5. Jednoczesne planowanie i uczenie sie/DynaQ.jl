
using Random

#actions coded as :left => 1, :down => 2, :right => 3, :up => 4
actions = Dict(1 => [0,-1], 2 => [1,0], 3 => [0,1], 4 => [-1,0]);

#arrows are corresponding to actions
arrows = Dict(1 => '⇐', 2 => '⇓', 3 => '⇒', 4 => '⇑');


rewards = Dict('S' => -0.05, 'G' => 1.0, 'H' => -1.0, 'F' => -0.05);

grid4x4= ['S' 'F' 'F' 'F';
        'F' 'H' 'F' 'H';
        'F' 'F' 'F' 'H';
        'H' 'F' 'F' 'G'];

grid8x8 =['S' 'F' 'F' 'F' 'F' 'F' 'F' 'F';
        'F' 'F' 'F' 'F' 'F' 'F' 'F' 'F';
        'F' 'F' 'F' 'H' 'F' 'F' 'F' 'F';
        'F' 'F' 'F' 'F' 'F' 'H' 'F' 'F';
        'F' 'F' 'F' 'H' 'F' 'F' 'F' 'F';
        'F' 'H' 'H' 'F' 'F' 'F' 'H' 'F';
        'F' 'H' 'F' 'F' 'H' 'F' 'H' 'F';
        'F' 'F' 'F' 'H' 'F' 'F' 'F' 'G';];



function get_grid(dim, p_holes, seed = 234)
    Random.seed!(seed)
    grid = [rand() < p_holes ? 'H' : 'F' for i in 1:dim, j in 1:dim]
    grid[1,1] = 'S'
    grid[end,end] = 'G'
    return grid
end


function reward_matrix(grid, rewards = rewards)
    R = zeros(size(grid))
    for i = 1:length(grid)
        R[i] = rewards[grid[i]]
    end
    return R
end

function get_policy(agent)
    P = rand(Int,size(agent.world))
    for i in CartesianIndices(P)
        P[i] = argmax([agent.Q[([i[1],i[2]], action)] for action in 1:length(agent.actions)])
    end
    P
end

function print_policy(P, grid, arrows = arrows)
    Policy = rand(Char,size(grid))
    for i = 1:length(grid)
        if grid[i] == 'F' || grid[i] == 'S' 
            Policy[i] = arrows[P[i]]
        elseif grid[i] == 'H' 
            Policy[i] = '⦷'
        else
            Policy[i] = grid[i]
        end
    end
    return Policy
end

function q_init(world,actions)
    Q = Dict{Tuple{Array{Int,1},Int}, Float64}()
    for s in eachindex(view(world, 1:size(world)[1], 1:size(world)[2]))
        for a in 1:length(actions)
            Q[[s[1],s[2]],a] = 0.0
        end
    end
    Q
end

#tworzymy obiekt agent, ktory przechowuje wszystkie kluczowe informacje na temat agenta
mutable struct Agent
    state::Array{Int64,1} #jego aktualna pozycja na planszy
    actions::Dict{Int, Array{Int,1}} #zbior ze wszystkimi potencjalnymi ruchami agenta: w gore, w dol, w lewo, w prawo
    ϵ::Float64 #współczynnik eksploracji
    β::Float64 #dyskonto
    α::Float64 #stopa uczenia się
    planning::Int64 #liczba krokow planowania
    Q::Dict{Tuple{Array{Int,1},Int}, Float64}
    model::Dict{Tuple{Array{Int,1},Int}, Tuple{Array{Int,1},Float64}}
    world::Array{Char,2} #swiat w ktorym toczy sie symulacja
    rewards::Array{Float64,2} #swiat w ktorym toczy sie symulacja
    score::Int #wynik - ile razy agent dotarl do mety
end

#konstruktor tworzacy agenta:
function Agent(grid, planning_steps; ϵ = 0.1, β = 0.99, α = 0.1, actions = actions, rewards = rewards) 
    Q = q_init(grid,actions)
    Agent([1,1], actions, 
        ϵ, β, α, 
        planning_steps, 
        Q, Dict{Tuple{Array{Int,1},Int}, Tuple{Array{Int,1},Float64}}(),
        grid, reward_matrix(grid, rewards), 0)
end

#funkcja ruchu
function go_in_direction(agent, state, action; deterministic = false)
    #wybieramy nowe pole:
    sides = filter(x -> !(x in [action, action .* -1]), agent.actions)
    p = rand()
    if deterministic
        new_state = state .+ actions[action]
    else
        if p <= 0.8
            new_state = state .+ actions[action]
        elseif p <= 0.9
            new_state = state .+ sides[1]
        else
            new_state = state .+ sides[2]
        end
    end
    #i jezeli miesci sie ono w wymiarach swiata przenosimy na nie agenta:
    if new_state[1] in (1:size(agent.world,1)) && new_state[2] in (1:size(agent.world,2))
        return new_state
    end
    return state
end

#funkcja uaktualniajaca wartosci macierzy Q
function learn!(agent, state, action, reward, new_state)
    oldv = agent.Q[(state, action)]
    if oldv == 0.0
        #jezeli agent nigdy nie byl na danym polu jego nagroda jest po prostu rowna nagrodzie ze znalezenie sie na nim:
        agent.Q[(state, action)] = reward
    else
        #jezeli pole bylo juz odwiedzone uaktualniamy jego wartosc
        qnext = maximum([agent.Q[(new_state, action)]  for action in keys(agent.actions)])
        agent.Q[(state, action)] +=  agent.α * (reward + agent.β*qnext - oldv)
    end
end

feed!(agent,action,reward,new_state)= (agent.model[(agent.state,action)] = (new_state, reward))

function dyna_Q!(agent, steps; burning = true, deterministic = false)
    agent.state = [1,1]
    step = 1.0
    episode = 0.0
    agent.ϵ = 1
    while step <= steps
        if (burning && step < 0.2*steps) || rand() < agent.ϵ
            action = rand(keys(agent.actions)) 
        else 
            action = argmax([agent.Q[(agent.state, action)] for action in 1:length(agent.actions)])
        end
        new_state = go_in_direction(agent, agent.state, action, deterministic = deterministic)
        reward = agent.rewards[new_state[1],new_state[2]]
        learn!(agent, agent.state, action, reward, new_state)
        feed!(agent,action,reward,new_state)
        for i = 1:agent.planning
            (state,action),(next_state,reward) = rand(agent.model)
            learn!(agent, state, action, reward, next_state)
        end 
        agent.world[new_state[1],new_state[2]] == 'G' && (agent.score += 1)
        if agent.world[new_state[1],new_state[2]] == 'H' || agent.world[new_state[1],new_state[2]] == 'G'
            agent.state = [1,1] 
            episode += 1.0
            agent.ϵ = 1.0/episode
        else
            agent.state = new_state
        end
        step += 1.0
    end
end

agent = Agent(grid4x4, 10);

dyna_Q!(agent, 10000.0, deterministic = true)

print_policy(get_policy(agent),grid4x4)

#using PyPlot

#res = []
#for i in 0:1:25
#    scores = []
#    for j in 1:10
#        agent = Agent(grid8x8, i);
#        dyna_Q!(agent, 10000.0, deterministic = true)
#        push!(scores,agent.score)
#    end
#    push!(res,sum(scores)/length(scores))
#end
#plot(res)


