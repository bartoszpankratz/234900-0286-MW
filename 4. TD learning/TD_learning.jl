
ENV["MPLBACKEND"]="qt4agg"
using PyPlot

function create_world(sizex = 4,sizey = 12)
    world = zeros(sizex,sizey)
    world[1,2:end-1] .= -1.0
    world[1,end] = 1.0
    return world
end

function q_init(world,actions)
    Q = Dict{Tuple{Array{Int64,1},Array{Int64,1}},Float64}()
    for s in eachindex(view(world, 1:size(world)[1], 1:size(world)[2]))
        for a in actions
            Q[[s[1],s[2]],a] = 0.0
        end
    end
    Q
end

#tworzymy obiekt agent, ktory przechowuje wszystkie kluczowe informacje na temat agenta
mutable struct Agent
    state::Array{Int64,1} #jego aktualna pozycja na planszy
    actions::Array{Array{Int64,1},1} #zbior ze wszystkimi potencjalnymi ruchami agenta: w gore, w dol, w lewo, w prawo
    last_action::Union{Nothing,Array{Int64,1}} #poprzednia akcja wybrana przez agenta
    last_state::Union{Nothing,Array{Int64,1}}  #jego poprzednia lokalizacja
    ϵ::Float64 #współczynnik eksploracji
    β::Float64 #dyskonto
    α::Float64 #stopa uczenia się
    Q::Dict{Tuple{Array{Int64,1},Array{Int64,1}},Float64} #slownik z nagrodami wynikajacymi z przejscia ze stanu do stanu
    world::Array{Float64,2} #swiat w ktorym toczy sie symulacja
    score::Int #wynik - ile razy agent dotarl do mety
    episodes::Int #liczba epizodow zakonczonych dotarciem do terminalnego stanu
    cumulated_reward::Int #Skumulowana nagroda agenta w czasie
    rewards::Array{Float64,1} #wektor przecietnego wyniku agenta w trakcie symulacji
end

#konstruktor tworzacy agenta:
function Agent(ϵ = 0.1,β = 0.99,α = 0.1) 
    actions = [[1,0],[0,1],[-1,0],[0,-1]]
    world = create_world()
    Q = q_init(world,actions)
    Agent([1,1], actions, 
        nothing, nothing, 
        ϵ, β, α, 
        Q, world,
        0, 0, 0, [])
end

#funkcja uczenia sie agenta - algorytm Q Learning
#uzupelniamy wartosc macierzy Q dla zadanego stanu s i akcji a
#robimy to poprzez dodanie do nagrody za znalezienie sie w tym stanie zdyskontowanej najwyzszej nagrody jaka moze 
#uzyskac agent przenoszac sie z tego stanu do nowego:
#Q(s, a) = reward(s) + beta * max(Q(s'))
function learn!(agent, reward)
    oldv = agent.Q[(agent.last_state, agent.last_action)]
    if oldv == 0.0
        agent.Q[(agent.last_state, agent.last_action)] = reward
    else
        maxqnew = maximum(agent.Q[(agent.state, action)]  for action in agent.actions)
        agent.Q[(agent.last_state, agent.last_action)] +=  agent.α * (reward + agent.β*maxqnew - oldv)
    end
end

#funkcja uczenia sie agenta - algorytm SARSA
#uzupelniamy wartosc macierzy Q dla zadanego stanu s i akcji a
#robimy to poprzez dodanie do nagrody za znalezienie sie w tym stanie zdyskontowanej najwyzszej nagrody jaka moze 
#uzyskac agent przenoszac sie z tego stanu do nowego:
#Q(s, a) = reward(s) + beta * Q(s',a')
function learn!(agent, reward, action)
    oldv = agent.Q[(agent.last_state, agent.last_action)]
    if oldv == 0.0
        agent.Q[(agent.last_state, agent.last_action)] = reward
    else
        qnew = agent.Q[(agent.state,action)]
        agent.Q[(agent.last_state, agent.last_action)] +=  agent.α * (reward + agent.β*qnew - oldv)
    end
end

#funkcja wyznaczajaca nagrode agenta za znalezienie sie na danym polu:
#jezeli to krawedz to nagroda to -100
#jezeli to meta to nagroda to 0
#jezeli "zwykle" pole nagroda to -1
function calculate_reward(agent)
    if agent.world[agent.state[1],agent.state[2]] == -1
        return -100
    elseif agent.world[agent.state[1],agent.state[2]] == 1
        agent.score += 1
        return 0
    else
        return -1
    end
end

#funkcja kontrolujaca wszystkie dzialania agenta
function update!(agent, burning,SARSA)
    #wyznaczamy nagrode ze znajdowania sie na danym polu:
    reward = calculate_reward(agent)
    #sprawdzamy czy symulacja jest na etapie "rozgrzewania"
    #jezeli tak agent losowo sie przemieszcza (eksploruje mape)
    #jezeli nie to zachlannie wybiera odpowiednia akcje
    if burning || rand() < agent.ϵ
        action = rand(agent.actions)
    else
        action_id = argmax([agent.Q[(agent.state, action)] for action in agent.actions])
        action = agent.actions[action_id]
    end
    #sprawdzamy czy agent moze sie uczyc, jezeli tak to uaktualniamy jego oczekiwania wobec stanow swiata:
    if !isa(agent.last_action, Nothing) && !isa(agent.last_state, Nothing)
        SARSA ? learn!(agent, reward, action) : learn!(agent, reward)
    end
    agent.last_state = agent.state
    agent.last_action = action
    #jezeli agent wypadl z mapy lub dotarl na mete przesuwamy go na start i zaczynamy od nowa:
    if agent.world[agent.state[1],agent.state[2]] == -1 || agent.world[agent.state[1],agent.state[2]] == 1
        agent.episodes += 1
        agent.cumulated_reward += reward
        push!(agent.rewards, agent.cumulated_reward/agent.episodes)
        agent.state = [1,1]
        agent.last_action = nothing
        agent.last_state = nothing
    else
        #jezeli nie to przesuwamy go na odpowiednie pole:
        new_state = agent.state .+ action
        #i jezeli miesci sie ono w wymiarach swiata przenosimy na nie agenta:
        if new_state[1] in (1:size(agent.world,1)) && new_state[2] in (1:size(agent.world,2))
            agent.state = new_state
        end
    end
end

#funkcja uaktualniajaca wykres
function plot!(agent, img, episode)
    field_val = agent.world[agent.state[1],agent.state[2]] 
    agent.world[agent.state[1],agent.state[2]] = 2
    img[:set_data](agent.world)  
    title("Episode: $episode, Score: $(agent.score)")
    show()         
    sleep(0.001)
    agent.world[agent.state[1],agent.state[2]] = field_val 
end

#funkcja kontrolujaca symulacje
function run_learning!(agent, episodes; plotting = true, burning = true, SARSA = false)
    step = 0
    if plotting
        field_val = agent.world[agent.state[1],agent.state[2]];
        agent.world[agent.state[1],agent.state[2]] = 2;
        img = imshow(agent.world);
        agent.world[agent.state[1],agent.state[2]] = field_val;
    end
    if burning
        while agent.episodes < 0.1 * episodes || step < 3000
            update!(agent, true, SARSA)
            if plotting
                plot!(agent, img, agent.episodes)
            end
            step += 1
        end
    end
    while agent.episodes < episodes
        update!(agent, false,SARSA)
        if plotting
            plot!(agent, img, agent.episodes)
        end
        step += 1
    end
    step
end

#uruchamiamy:

agent_Q = Agent();

run_learning!(agent_Q,5000,plotting = false, SARSA = false)

agent_SARSA = Agent();

run_learning!(agent_SARSA,5000,plotting = false, SARSA = true)


