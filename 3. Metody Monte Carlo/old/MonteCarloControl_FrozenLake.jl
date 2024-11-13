using ReinforcementLearning, Plots, Random

#actions coded as :left => 1, :down => 2, :right => 3, :up => 4
#arrows are corresponding to actions
arrows = Dict(1 => '⇐', 2 => '⇓', 3 => '⇒', 4 => '⇑');

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

#Auxilliary functions
function get_grid(dim, p_holes, seed = 234)
    Random.seed!(seed)
    grid = [rand() < p_holes ? 'H' : 'F' for i in 1:dim, j in 1:dim]
    grid[1,1] = 'S'
    grid[end,end] = 'G'
    return grid
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


mutable struct FrozenLakeEnv <: AbstractEnv
    reward::Union{Nothing, Float64}
    rewards::Dict{Char, Float64} 
    actions::Dict{Int64, Tuple{Int64, Int64}}  
    world::Matrix{Char} 
    ES::Bool 
    position::Int 
end
Main.FrozenLakeEnv

function FrozenLakeEnv(grid::Union{Int,Symbol} = :grid4x4; ES::Bool = false) 
    if typeof(grid) == Int
        world = get_grid(grid, 0.1)
    elseif grid == :grid4x4
        world = grid4x4
    elseif grid == :grid8x8
        world = grid8x8
    else
        @error "unknown grid"
    end
    ES == true ? position = rand(1:length(world)) : position = 1
    return FrozenLakeEnv(nothing, Dict('S' => -0.05, 'G' => 1.0, 'H' => -1.0, 'F' => -0.05),
    Dict(1 => (0,-1), 2 => (1,0), 3 => (0,1), 4 => (-1,0)), 
        world, ES, position)
    
end

#functions:
RLBase.action_space(env::FrozenLakeEnv) = Base.OneTo(length(env.actions))
RLBase.state_space(env::FrozenLakeEnv) = Base.OneTo(length(env.world))
RLBase.reward(env::FrozenLakeEnv) = env.reward
RLBase.state(env::FrozenLakeEnv) = env.position
RLBase.is_terminated(env::FrozenLakeEnv) = env.reward == 1.0 || env.reward == -1.0
function RLBase.reset!(env::FrozenLakeEnv) 
    env.reward = nothing
    env.ES == true ? env.position = rand(state_space(env)) : env.position = 1
end


function (x::FrozenLakeEnv)(action)
    @assert action in action_space(x)
    direction = x.actions[action]
    cartesian_state = (CartesianIndices(x.world)[x.position][1],
                        CartesianIndices(x.world)[x.position][2])
    sides = filter(y -> !(y in [direction, direction .* -1]), collect(values(x.actions)))
    p = rand()
    if p <= 0.8
        new_state = cartesian_state .+ direction
    elseif p <= 0.9
        new_state = cartesian_state .+ sides[1]
    else
        new_state = cartesian_state .+ sides[2]
    end
    if new_state[1] in (1:size(x.world,1)) && new_state[2] in (1:size(x.world,2))
        x.position = LinearIndices(x.world)[new_state[1],new_state[2]]
        x.reward = x.rewards[x.world[x.position]]
    else
        x.position = LinearIndices(x.world)[cartesian_state[1],cartesian_state[2]]
        x.reward = x.rewards[x.world[x.position]]
    end
end

env = FrozenLakeEnv(:grid8x8, ES = false);
RLBase.test_runnable!(env);

hook = TotalRewardPerEpisode()
TotalRewardPerEpisode(Float64[], 0.0, true)

run(RandomPolicy(action_space(env)), env, StopAfterEpisode(200_000), hook)

mcpolicy = QBasedPolicy(
           learner = MonteCarloLearner(
                   approximator=TabularQApproximator(
                       ;n_state = length(state_space(env)),
                       n_action = length(action_space(env)),
                        init = 0.0,
                   ),
            γ = 0.999
               ),
           explorer = EpsilonGreedyExplorer(0.1)
       )

run(mcpolicy, env, StopAfterEpisode(200_000), hook)

agentMC = Agent(
           policy = mcpolicy,
           trajectory = VectorSARTTrajectory()
       )

run(agentMC, env, StopAfterEpisode(200_000), hook)

mutable struct AgentMC
    env::FrozenLakeEnv
    ϵ::Float64
    β::Float64 #stopa dyskonta
    Q::Array{Float64,2}#macierz wartosci dla kazdej z par:stan,akcja
    C::Array{Float64,2} #macierz wystapien dla kazdej z par:stan,akcja
    π::Array{Int} #strategia agenta
end

function AgentMC(env; ϵ = .2, β = 0.999)
    return AgentMC(env,ϵ, β,
        zeros(length(env.world), length(action_space(env))), 
        zeros(length(env.world), length(action_space(env))),
        rand(1:length(action_space(env)),length(env.world)))
end

function get_episode(agent, π = agent.π; maxstep = 1000)
    step = 1
    if agent.env.ES == true 
        S = rand(state_space(agent.env)) 
		agent.env.position = S
        action = rand(action_space(agent.env))
    else
        S = 1
        rand() < agent.ϵ ? action = rand(action_space(agent.env)) : action = π[S]
    end
    agent.env(action)
    r = agent.env.reward
    episode = [(S, action, r)]
    if is_terminated(agent.env) 
        reset!(agent.env)
        return episode
    end
    while step < maxstep
        S = state(agent.env)
        (rand() < agent.ϵ) && (agent.env.ES == false) ? action = rand(action_space(agent.env)) : action = π[S]
        agent.env(action)
        r = agent.env.reward
        push!(episode, (S, action,r))
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

#xploring starts 
agent = AgentMC(FrozenLakeEnv(:grid4x4, ES = true));
MC!(agent, maxit = 500_000)
print_policy(agent.π, agent.env.world)

#without exploring starts
agent = AgentMC(FrozenLakeEnv(:grid4x4, ES = false));
success_rate = MC!(agent; maxit = 500_000)
print_policy(agent.π, agent.env.world)

plot(success_rate[2:end], xlabel = "Iteration", ylabel = "success rate", legend = false)
