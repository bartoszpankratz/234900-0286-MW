
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

function random_policy(grid,actions = actions)
    P = rand(Int,size(grid))
    for i = 1:length(grid)
        P[i] = rand(1:length(actions))
    end
    return P
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

#tworzymy obiekt agent, ktory przechowuje wszystkie kluczowe informacje na temat agenta
mutable struct Agent
    actions::Dict{Int, Array{Int,1}} #zbior ze wszystkimi potencjalnymi ruchami agenta: w gore, w dol, w lewo, w prawo
    ϵ::Float64 #współczynnik eksploracji
    β::Float64 #dyskonto
    Q::Array{Float64,2}#macierz wartosci dla kazdej z par:stan,akcja
    C::Array{Float64,2} #macierz wystapien dla kazdej z par:stan,akcja
    world::Array{Char,2} #swiat w ktorym toczy sie symulacja
    π::Array{Int,2} #strategia agenta
    rewards::Array{Float64,2} #macierz nagrod w zaleznosci od stanu w ktorym znajduje sie agent
end

#konstruktor agenta
function Agent(grid; ϵ = .2, β = 0.999,  actions = actions, rewards = rewards)
    return Agent(actions,ϵ, β,
        zeros(length(grid), length(actions)), zeros(length(grid), length(actions)),
        grid, random_policy(grid),reward_matrix(grid))
end

#funkcja ruchu
function goInDirection(agent, state, action)
    #wybieramy nowe pole:
    state_cart = [CartesianIndices(agent.world)[state][1], CartesianIndices(agent.world)[state][2]]
    sides = filter(x -> !(x in [action, action .* -1]), agent.actions)
    p = rand()
    if p <= 0.8
        new_state = state_cart .+ actions[action]
    elseif p <= 0.9
        new_state = state_cart .+ sides[1]
    else
        new_state = state_cart .+ sides[2]
    end
    #i jezeli miesci sie ono w wymiarach swiata przenosimy na nie agenta:
    if new_state[1] in (1:size(agent.world,1)) && new_state[2] in (1:size(agent.world,2))
        return LinearIndices(agent.world)[new_state[1],new_state[2]]
    end
    return state
end

function get_episode(agent, π = agent.π; maxstep = 1000)
    step = 1
    state = 1
    rand() < agent.ϵ ? (action = rand(keys(actions))) : (action = π[state])
    episode = [(state, action)]
    (agent.world[state] == 'H' || agent.world[state] == 'G' ) && return episode
    while step < maxstep
        state, action = episode[end]
        new_state = goInDirection(agent, state, action)
        rand() < agent.ϵ ? (new_action = rand(keys(actions))) : (new_action = π[new_state])
        push!(episode, (new_state, new_action))
        (agent.world[new_state] == 'H' || agent.world[new_state] == 'G' ) && break
        step +=1
    end
    return episode
end

# On-Policy Monte Carlo control

function update!(agent)
    episode = get_episode(agent)
    R = 0
    occur_first = reverse(vcat(findfirst.(isequal.(unique(episode)), [episode]), length(episode) + 1))
    for k = 2:length(occur_first)
        for i = occur_first[k-1] - 1:-1:occur_first[k]
            state,action = episode[i]
            R = agent.β*R + agent.rewards[state]
        end
        state,action = episode[occur_first[k]]
        agent.C[state,action] += 1
        agent.Q[state,action] += (R - agent.Q[state,action])/ agent.C[state,action]
        agent.π[state] = argmax(agent.Q[state,:])   
    end  
end

function MC!(agent; maxit = 100000)
    iter = 0
    while iter < maxit
        update!(agent)
        iter +=1
    end
end

agent = Agent(grid8x8, ϵ = 1/3);


@time MC!(agent, maxit = 700_000)

print_policy(agent.π,agent.world)


