using Plots, LinearAlgebra, ReinforcementLearning, IntervalSets
using Flux
using Flux: params
using Plots; 
gr()
import StatsBase.sample

#Balls object - red ones (type "0") are poisonous and green ones (type "1") are edible 
mutable struct Ball{TI<:Integer, TF<:AbstractFloat}
    kind::TI #1 poison, 2 food
    loc::NamedTuple{(:x, :y), Tuple{TF, TF}}
    radius::Float64
    age::TI
end


#balls constructor
Ball(k) = k ∈ [1,2] ? Ball(k,(x =rand(), y = rand()),0.05, rand(1:100)) : @error "wrong type of object - it could be 1 or 2!"

#definition of agent's eye:
mutable struct Eye{TI<:Integer, TF<:AbstractFloat}
    angle::TF
    max_range::TF
    sensed_proximity::TF
    sensed_type::TI #0 nothing; 1 poison, 2 food, 3 wall
end

Eye(a) = Eye(a, 0.2,0.2, 0)

mutable struct ContinuousLabirynthEnv <: AbstractEnv
    walls::Vector{Vector{NamedTuple{(:x, :y), Tuple{Float64, Float64}}}}
    nb::Int64
    balls::Vector{Ball{Int64, Float64}}
    observation_space::Space{Vector{ClosedInterval{Float64}}}
    action_space::Space{Vector{ClosedInterval{Float64}}}
    velocity::Vector{Float64}
    old_position::NamedTuple{(:x, :y), Tuple{Float64, Float64}}
    position::NamedTuple{(:x, :y), Tuple{Float64, Float64}}
    radius::Float64
    angle::Float64
    eyes::Vector{Eye{Int64, Float64}}
    digestion_signal::Float64
end
Main.ContinuousLabirynthEnv

function ContinuousLabirynthEnv(nb; radius = 0.05)
    walls = [#bounds of the map:
        [(x = 0.0, y = 0.0),(x = 0.0, y = 1.0)],
        [(x = 0.0, y = 0.0),(x = 1.0, y = 0.0)],
        [(x = 1.0, y = 0.0),(x = 1.0, y = 1.0)],
        [(x = 0.0, y = 1.0),(x = 1.0, y = 1.0)],
        #walls inside the map:
        [(x = 0.1, y = 0.1),(x = 0.3, y = 0.1)],
        [(x = 0.1, y = 0.9),(x = 0.3, y = 0.9)],
        [(x = 0.3, y = 0.1),(x = 0.3, y = 0.9)],
        
        [(x = 0.7, y = 0.1),(x = 0.9, y = 0.1)],
        [(x = 0.7, y = 0.9),(x = 0.9, y = 0.9)],
        #[(x = 0.9, y = 0.1),(x = 0.9, y = 0.9)], #can be use instead of the latter one:
        [(x = 0.7, y = 0.1),(x = 0.7, y = 0.9)]
        ]
    balls = [Ball(rand([1,2])) for i = 1:nb]
    position = (x = rand(), y = rand());
    eyes = [Eye((k - 4)*0.25) for k = 1:9];
    observation_space = Space([0.0..1.0 for i = 1:(3 * length(eyes))])
    action_space = Space([0.0..1.0, 0.0..1.0])
    return ContinuousLabirynthEnv(walls, nb, balls, observation_space, action_space, [0.0, 0.0],
            position, position, radius, 0.0, eyes, 0.0)
end

#rotate a vector clockwise https://en.wikipedia.org/wiki/Rotation_matrix

rot(vec, angle) = [cos(angle) sin(angle); -sin(angle) cos(angle)] * vec

#line circle intersect: 
#https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
function intersect(vec,centre,radius)
    v = values(vec[2]) .- values(vec[1])
    a = sum(v.*v)
    b = 2* sum(v .* (values(vec[1]) .- values(centre)))
    c =  sum(values(vec[1]) .* values(vec[1])) + sum(values(centre) 
        .* values(centre)) - 2 * sum(values(vec[1]) .* values(centre)) - radius ^2
    Δ = b^2 - 4*a*c
    if Δ < 0
        return false
    else
        t₁ = (-b + √Δ)/(2*a)
        t₂ = (-b - √Δ)/(2*a)
        if 0.0 ≤ t₁ ≤ 1 || 0.0 ≤ t₂ ≤ 1
            p₁= values(vec[1]) .+ t₁ .* v
            p₂ = values(vec[1]) .+ t₂ .* v
            norm(p₁ .- values(vec[1])) < norm(p₂ .- values(vec[1])) ? (return p₁) : (return p₂)
        else
            return false
        end 
    end
end

#interscetion of two lines:
#http://www-cs.ccny.cuny.edu/~wolberg/capstone/intersection/Intersection%20point%20of%20two%20lines.html
function intersect(vec1,vec2)
    denominator = (vec2[2].y - vec2[1].y)*(vec1[2].x - vec1[1].x) - (vec2[2].x - vec2[1].x)*(vec1[2].y - vec1[1].y)
    denominator == 0.0 && (return false)
    u₁ = ((vec2[2].x - vec2[1].x)*(vec1[1].y - vec2[1].y) - (vec2[2].y - vec2[1].y)*(vec1[1].x - vec2[1].x)) / denominator
    u₂ = ((vec1[2].x - vec1[1].x)*(vec1[1].y - vec2[1].y) - (vec1[2].y - vec1[1].y)*(vec1[1].x - vec2[1].x)) / denominator
    (0.0 ≤ u₁ ≤ 1.0 &&  0.0 ≤ u₂ ≤ 1.0) ? (return (vec1[1].x + u₁ * (vec1[2].x - vec1[1].x),
                                                    vec1[1].y + u₁ * (vec1[2].y - vec1[1].y))) : (return false) 
end




#vector of agent's eyesight:
eyesight(env, eye) = [env.position, (x = env.position.x + sin(env.angle + eye.angle)*eye.sensed_proximity, 
                            y = env.position.y + cos(env.angle + eye.angle)*eye.sensed_proximity)]

eyesight(env) = [eyesight(env,eye) for eye in env.eyes]

#plotting 
function plot(env::ContinuousLabirynthEnv)
    p = Plots.plot(framestyle = :none, legend=:none)
    
    for wall in env.walls
        plot!([w.x for w in wall],[w.y for w in wall], linewidth=2, c = :blue)
    end
    
    scatter!([ball.loc.x for ball in env.balls if ball.kind == 1], 
            [ball.loc.y for ball in env.balls if ball.kind == 1], 
            color=:red, m = :circle, markersize=16, alpha=0.6)

    scatter!([ball.loc.x for ball in env.balls if ball.kind == 2], 
            [ball.loc.y for ball in env.balls if ball.kind == 2], 
            color=:green, m = :circle, markersize=16, alpha=0.6)
    
    scatter!([env.position.x,], [env.position.y,], color=:orange, markersize=20, alpha=0.8)

    for eye in eyesight(env) 
        plot!([e.x for e in eye],[e.y for e in eye], linewidth=0.5, c = :black,alpha=0.6)
    end
    display(p)
end

#functions:
RLBase.action_space(env::ContinuousLabirynthEnv) = env.action_space

"""
    state_space(env::ContinuousLabirynthEnv)

State is represented as  vector three times number of eyes and is coded as follows:
-every three neighboring values are representing a signal visible by one eye, e.g. 
 [1.0, 1.0, 0.14] means that agent do not see any edible balls (first value of vector)
nor poisons (second value of vector), but see the wall in the proximity of 0.14
"""
RLBase.state_space(env::ContinuousLabirynthEnv) = env.observation_space

function RLBase.state(env::ContinuousLabirynthEnv) 
    state = ones(Float64, length(env.observation_space))
    for (i, eye) in enumerate(env.eyes)
        eye.sensed_type == 0 && continue 
        state[(i-1)*3 + eye.sensed_type] = eye.sensed_proximity/eye.max_range
    end
    return state
end

function RLBase.reward(env::ContinuousLabirynthEnv)
    #agent do not like to see walls, especially up close:
    proximity_reward = 0.0
    for eye in env.eyes
        if eye.sensed_type == 3
            proximity_reward -= 1 - eye.sensed_proximity/eye.max_range
        elseif eye.sensed_type != 0
            proximity_reward += 1 - eye.sensed_proximity/eye.max_range
        end
    end
    #agent like to go forward:
    forward_reward = 0.0
    if env.old_position == env.position 
        forward_reward += -1.0
    else
        forward_reward += 1.0
    end
    ##agent also like to eat good things:
    return env.digestion_signal + proximity_reward + forward_reward
end

RLBase.is_terminated(env::ContinuousLabirynthEnv) = false
function RLBase.reset!(env::ContinuousLabirynthEnv) 
    env.position = (x = rand(), y = rand());
    env.angle = 0.0
    env.digestion_signal = 0.0
end

#check if agent do not fall from the map:
on_boundary(vec) = any(vec .≤ 0.0) || any(vec .≥ 1.0)

#check if agent do not collide with a wall:
function collide(env, vec)
    for wall in env.walls
        if intersect(wall,vec,env.radius) != false 
            env.angle  += π/2 #we turn agent 180° to avoid him stucking in the wall 
            return true 
        end
    end
    return false
end

#movement function of agent
function turnaround_and_move!(env, action)
    #agent's movement is caused by the two  wheels rotating with different velocities and directions (action) 
    #wheels are perpendicular to the direction agent is facing at the moment
    
    #firstly we must find the positions of the wheels
    #we will rotate agent's radius by 90°:
    vec = rot([0.0, env.radius],env.angle + π/2)
    #now, we could find the positions of both wheels:
    wheel_1 = [env.position.x + vec[1], env.position.y + vec[2]] 
    wheel_2 = [env.position.x - vec[1], env.position.y - vec[2]] 
    #rotate first wheel, clockwise:
    vel_clockwise = rot(-1 .* vec, action.v1)
    #rotate second wheel, counterclockwise:
    vel_counterclockwise = rot(vec, -action.v2)
    #new position of first wheel:
    wheel_1 = [wheel_1[1] + vel_clockwise[1], wheel_1[2] + vel_clockwise[2]]
    #new position of second wheel:
    wheel_2 = [wheel_2[1] + vel_counterclockwise[1], wheel_2[2] + vel_counterclockwise[2]]
    #new position of agent is just an average of the positions of both wheels:
    new_position = [wheel_1[1] + wheel_2[1], wheel_1[2] + wheel_2[2]] ./ 2
    
    #now we must adjust the angle that agent is facing:
    env.angle += action.v1
    env.angle > 2*π && (env.angle -= 2 * π)
    env.angle -= action.v2
    env.angle < 0 && (env.angle += 2 * π)
    
    #and check if new position is feasible:
    if !(collide(env, new_position) || on_boundary(new_position))
        env.old_position = deepcopy(env.position)
        env.position  = (x = new_position[1], y = new_position[2])
   end 
end

#uaktualniamy obiekty, ktore widzi agent:
#dla kazdego oka sprawdzamy czy widzi jakis objekt lub sciane, jezeli tak to w jakiej odleglosci
#dodatkowo patrzymy czy agent nic nie zjadl - nie przecial sie z kulka oznaczajaca jedzenie
function look_at_things_and_eat!(env)
    for eye in env.eyes
        eye.sensed_proximity = eye.max_range
        eye.sensed_type = 0
        eye_sight = eyesight(env,eye)
        for wall in env.walls
            intersect_point = intersect(eye_sight,wall)
            intersect_point == false && continue 
            proximity = norm(values(env.position) .- intersect_point)
            if proximity < eye.sensed_proximity
                eye.sensed_proximity = proximity
                eye.sensed_type = 3
            end
        end
        for ball in env.balls
            intersect_point = intersect(eye_sight,ball.loc, ball.radius)
            intersect_point == false && continue 
            proximity = norm(values(env.position) .- intersect_point)
            if proximity < eye.sensed_proximity
                if proximity < (env.radius + ball.radius)
                    ball.kind == 2 ? (env.digestion_signal = 20.0) : (env.digestion_signal = -30.0)
                    ball.age = 9999999999999 
                else
                    eye.sensed_proximity = proximity
                    eye.sensed_type = ball.kind
                end
            end 
        end
    end
end


function update_balls!(env)
    for ball in env.balls
        ball.age += 1
    end
    env.balls = filter(ball -> ball.age < 500, env.balls)
    for new_ball in 1:(env.nb - length(env.balls))
        push!(env.balls, Ball(rand([1,2])))
    end    
end

function (x::ContinuousLabirynthEnv)(action)
    @assert action in action_space(x)
    x.velocity = [action.v1, action.v2]
    x.digestion_signal = 0.0
    turnaround_and_move!(x, action)
    look_at_things_and_eat!(x)
    update_balls!(x)
end

mutable struct Brain
    experience_size::Int64
    experience::Array
    min_experience_size::Int64
    net::Flux.Chain
    μ::Dense
    logσ::Dense
    value_net::Flux.Chain
    ηₚ::Float64
    ηᵥ::Float64
    β::Float64
    batch_size::Int64
end

function Brain(input_size; experience_size = 3000, min_memory_size = 1000, ηₚ = .01, ηᵥ = 0.001, β = 0.95, batch_size = 64)
    net = Chain(Dense(input_size,128,relu),Dense(128, 64,relu),
        Dense(64, 32, identity))
    μ = Dense(32, 2, sigmoid)
    logσ = Dense(32, 2, sigmoid)
    value_net = Chain(Dense(input_size,128,identity),
        Dense(128, 64,relu),Dense(64, 32,relu),Dense(32, 1,identity));
    return Brain(experience_size, [], min_memory_size, net, μ, logσ, value_net, ηₚ, ηᵥ, β, batch_size)
end

mutable struct ContinuousAgent
    env::ContinuousLabirynthEnv 
    brain::Brain
end

function ContinuousAgent(nb)
    env = ContinuousLabirynthEnv(nb)
    brain = Brain(length(state_space(env)))
    return ContinuousAgent(env,brain)
end

function forward(brain, state)
    v = brain.value_net(state)
    Z = brain.net(state)
    mu = brain.μ(Z)
    logsigma = brain.logσ(Z)
    return v[1], mu, logsigma
end

function remember!(brain::Brain, step::Tuple)
    length(brain.experience) == brain.experience_size && deleteat!(brain.experience,1)
    push!(brain.experience, step)
end

critic_loss(x, y, ξ = 0.5) = ξ*Flux.mse(agent.brain.value_net(x), y)


function gaussian_loss(s, a, v, γ = 0.001)
    Z = agent.brain.net(s)
    mu = agent.brain.μ(Z)
    sigma = exp.(agent.brain.logσ(Z))
    pdf_val = 1 ./ (sigma .* sqrt(2*π)) .* exp.(-0.5 .* ((a .- mu) ./ sigma).^2)
    log_prob = log.(pdf_val .+ 1e-7)
    entropy = sum(-log.(pdf_val .+ 1e-7) .* pdf_val) / length(log_prob)
    return sum(-log_prob .* v) / length(log_prob) - γ * entropy
end

function replay!(agent::ContinuousAgent)
    S = zeros(Float32,length(state(agent.env)), agent.brain.batch_size)
    A = zeros(Float32,length(action_space(agent.env)), agent.brain.batch_size)
    Adv = zeros(Float32, 1, agent.brain.batch_size)
    V = zeros(Float32,1, agent.brain.batch_size)
    for (i,step)  in enumerate(sample(agent.brain.experience, agent.brain.batch_size, replace = false))
        s,a,r,s′,v,v′ = step
        R = r + agent.brain.β * v′
        adv = R - v
        S[:, i] .= s
        A[:, i] .= a
        Adv[:, i] .= adv
        V[:, i] .= R
    end
    
    Flux.train!(gaussian_loss, params(agent.brain.net, agent.brain.μ, agent.brain.logσ), [(S,A,Adv)], ADAM(agent.brain.ηₚ))
    Flux.train!(critic_loss, params(agent.brain.value_net), [(S, V)], ADAM(agent.brain.ηᵥ))
    #push!(agent.losses, gaussian_loss(S,A,R))
end

function step!(agent::ContinuousAgent,  training::Bool)
    s = deepcopy(state(agent.env))
    v, μ,logσ = forward(agent.brain, s)
    a = μ + exp.(logσ) .* randn(length(logσ))
    a = (Flux.tanh.(a) .+1 ) ./ 2
    agent.env((v1 = a[1], v2 = a[2]))
    r = deepcopy(reward(agent.env))
    s′ = deepcopy(state(agent.env))
    v′,_ ,_ = forward(agent.brain, s′)
    remember!(agent.brain, (s,a,r,s′,v,v′))
    (training && length(agent.brain.experience) > agent.brain.min_experience_size) && replay!(agent)
end

function run!(agent::ContinuousAgent, steps::Int; training::Bool = true,
            plotting::Bool = true, summary::Bool = true)
    step = 1.0
    while step ≤ steps
        plotting && (plot(agent.env); sleep(0.01))
        step!(agent, training)
        if summary && mod(step,5000) == 0
            @info "step $(Int(step))"
            @info "Reward: $(reward(agent.env))"
            s,a,r,s′,v,v′ = agent.brain.experience[end]
            @info "actor loss: $(gaussian_loss(s,a, (r + agent.brain.β *  v′) - v))"
            @info "critc loss: $(critic_loss(s,r + agent.brain.β *  v′))"
        end
        step += 1.0
    end
end


agent = ContinuousAgent(50)
run!(agent,1_000_000, plotting = false, summary = true)

s1 = ones(Float64,27)
s1[1] = 0.13
    v, μ,logσ = forward(agent.brain, s1)
    a = μ + exp.(logσ) .* randn(length(logσ))
    a = (Flux.tanh.(a) .+1 ) ./ 2






#run!(agent,1000, plotting = true,training = false, summary = true)
