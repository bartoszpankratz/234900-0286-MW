
#Modified frozen lake example, see: https://gym.openai.com/envs/FrozenLake-v0/

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



#Transition matrix T(s',a,s) - from state s to successor state s' by taking action a 
#assuming that ice is slippery and probability of moving forward is equal to 0.8, 
#probabilities of moving left or right are equal to 0.1

function transition_matrix(grid, actions = actions)
    T = zeros(length(grid),length(actions),length(grid))
    i2s = CartesianIndices(grid)
    s2i = LinearIndices(grid)
    for i = 1:length(grid)
        if !(grid[i] == 'H' || grid[i] == 'G' )
            index = i2s[i]
            for j = 1:length(actions)
                indices = Tuple(index) .+ actions[j]
                if all(in.( indices, (1:size(grid,1), 1:size(grid,2)))) 
                    k = s2i[indices...]
                    T[k,j,i] += 0.8
                    if actions[j][1] == 0
                        for l in [-1,1]
                            ind = Tuple(index) .+ (l,0)
                            if all(in.( ind, (1:size(grid,1), 1:size(grid,2)))) 
                                T[s2i[ind...],j,i] += 0.1
                            else
                                T[i,j,i] += 0.1
                            end
                        end
                    else
                        for l in [-1,1]
                            ind = Tuple(index) .+ (0,l)
                            if all(in.( ind, (1:size(grid,1), 1:size(grid,2))))
                                T[s2i[ind...],j,i] += 0.1
                            else
                                T[i,j,i] += 0.1
                            end
                        end
                    end
                else
                    T[i,j,i] += 0.8
                    if actions[j][1] == 0
                        for l in [-1,1]
                            ind = Tuple(index) .+ (l,0)
                            if all(in.( ind, (1:size(grid,1), 1:size(grid,2)))) 
                                T[s2i[ind...],j,i] += 0.1
                            else
                                T[i,j,i] += 0.1
                            end
                        end
                    else
                        for l in [-1,1]
                            ind = Tuple(index) .+ (0,l)
                            if all(in.( ind, (1:size(grid,1), 1:size(grid,2)))) 
                                T[s2i[ind...],j,i] += 0.1
                            else
                                T[i,j,i] += 0.1
                            end
                        end
                    end
                end
            end
        end
    end
    return T
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

#policy evaluation

function evaluate!(P, v, R, T, β)
    for s = 1:length(v)
        v[s]= R[s] + β * sum(v .*  T[:,P[s],s])
    end
end 

function evaluate_policy(grid,P;
                        β = 0.999, ϵ=0.0001, 
                        actions = actions)
    iter = 0
    T = transition_matrix(grid)
    R = reward_matrix(grid)
    v₁ = zeros(length(grid))
    while true
        iter += 1
        v = deepcopy(v₁)
        evaluate!(P, v₁, R, T, β)
        @info v₁
        δ = maximum(abs.(v₁ - v)) 
        δ < ϵ * (1 - β) / β && break 
    end 
    
    println("Iterations: $(iter)")
    return reshape(v₁,size(grid)),  print_policy(P, grid)
end

#value iteration algorithm

function get_policy(v, T, actions = actions)
    P = rand(Int,length(v))
    for s = 1:length(v)
        actions_vector = zeros(length(actions))
        for i = 1:length(actions)
            actions_vector[i] = sum(v .* T[:,i,s])
        end
        P[s] = argmax(actions_vector)
    end 
    return P
end

function update_values!(v₁, v, T,R,β, actions = actions)
    for s = 1:length(v₁)
        actions_vector = zeros(length(actions))
        for i = 1:length(actions)
            actions_vector[i] = sum(v .* T[:,i,s])
        end
        v₁[s] = R[s] +  β * maximum(actions_vector)
    end
end

 function value_iteration(grid,β = 0.999, ϵ=0.0001, actions = actions)
    iter = 0
    T = transition_matrix(grid)
    R = reward_matrix(grid)
    v₁ = zeros(length(grid))
    while true
        iter += 1
        v = deepcopy(v₁)
        update_values!(v₁,v, T,R,β)
        δ = maximum( abs.(v₁ - v))  
        δ < ϵ * (1 - β) / β && break
    end
    P = get_policy(v₁, T)
    println("Iterations: $(iter)")
    return reshape(v₁,(size(grid))), print_policy(P, grid)
end


vᵥ, pᵥ =  value_iteration(grid8x8);

#policy iteration algorithm

function evaluate!(P, v, v₁, R, T, β)
    for s = 1:length(v)
        v₁[s]= R[s] + β * sum(v .*  T[:,P[s],s])
    end
end 

function improve_policy!(v, T, P, actions = actions)
    for s = 1:length(v)
        actions_vector = zeros(length(actions))
        for i = 1:length(actions)
            actions_vector[i] = sum(v .* T[:,i,s])
        end
        action = argmax(actions_vector)
        action != P[s] && (P[s] = action)
    end
end

function policy_iteration(grid,β = 0.999, ϵ=0.0001)
    iter = 1
    T = transition_matrix(grid)
    R = reward_matrix(grid)
    v₁ = zeros(length(grid))
    P = random_policy(grid)
    while true
        iter += 1
        v = deepcopy(v₁)
        evaluate!(P, v₁, R, T, β)
        δ = maximum( abs.(v₁ - v)) 
        δ < ϵ * (1 - β) / β && break 
        improve_policy!(v₁, T, P)
    end 
    println("Iterations: $(iter)")
    return reshape(v₁,size(grid)),  print_policy(P, grid)
end

 vₚ, pₚ = policy_iteration(grid8x8);
