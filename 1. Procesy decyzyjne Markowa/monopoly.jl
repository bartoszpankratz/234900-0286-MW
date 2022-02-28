using DataFrames
using Random

            Spaces = [
                ("Go", 0.0, Inf),
                ("Mediterranean Avenue", 2.0, 60.0),
                ("Community Chest", 0.0, Inf),
                ("Baltic Avenue", 4.0, 60.0),
                ("Income Tax", 0.0, Inf),
                ("Reading Railroad", 25.0, 200.0),
                ("Oriental Avenue", 6.0, 100.0),
                ("Chance", 0.0, Inf),
                ("Vermont Avenue", 6.0, 100.0),
                ("Connecticut Avenue", 8.0, 120.0),
                ("Jail", 0.0, Inf),
                ("St. Charles Place", 10.0, 140.0),
                ("Electric Company", 4.0 * 6.0, 150.0),
                ("States Avenue", 10.0, 140.0),
                ("Virginia Avenue", 12.0, 160.0),
                ("Pennsylvania Railroad", 25.0, 200.0),
                ("St. James Place", 14.0, 180.0),
                ("Community Chest", 0.0, Inf),
                ("Tennessee Avenue", 14.0, 180.0),
                ("New York Avenue", 16.0, 200.0),
                ("Free Parking", 0.0, Inf),
                ("Kentucky Avenue", 18.0, 220.0),
                ("Chance", 0.0, Inf),
                ("Indiana Avenue", 18.0, 220.0),
                ("Illinois Avenue", 20.0, 240.0),
                ("B & O Railroad", 25.0, 200.0),
                ("Atlantic Avenue", 22.0, 260.0),
                ("Ventnor Avenue", 22.0, 260.0),
                ("Water Works", 4.0 * 6.0, 150.0),
                ("Marvin Gardens", 24.0, 280.0),
                ("Go To Jail", 0.0, Inf),
                ("Pacific Avenue", 26.0, 300.0),
                ("North Carolina Avenue", 26.0, 300.0),
                ("Community Chest", 0.0, Inf),
                ("Pennsylvania Avenue", 28.0, 320.0),
                ("Short Line", 25.0, 200.0),
                ("Chance", 0.0, Inf),
                ("Park Place", 35.0, 350.0),
                ("Luxury Tax", 0.0, Inf),
                ("Boardwalk", 50.0, 400.0)];

function transition_matrix(n=40)
    #basic transition matrix
    T = zeros(Float64,n,n)
    for i = 1:n
        for j = 2:12
            T[i, mod(i + j - 1,n)+1] = mod(min(j -1,13-j),7)/ 36
        end
    end
    #special matrix
    S = zeros(Float64,n,n)
    for i = 1:n
        if !in(i,(3,18,31,34))
            S[i,i] = 1.0
        elseif i == 31
            S[i,11] = 1.0
        else #community chest
            #advance to go
            S[i,1] = 1.0 / 16.0;
            #go to jail
            S[i,11] = 1.0 / 16.0;
            #stay put
            S[i,i] = 14.0 / 16.0;
        end
    end
    T * S
end

function solve(k,n=40)
    T = transition_matrix(n)
    s = hcat(1.0,zeros(Float64,1,n-1))
    s * T^k
end

#first as a simple Markov chain:
solve(100000,40)'

#add some rewards:
function summary(k = 100000, n=40)
    probs = solve(k,n)
    df = DataFrame(Space = String[], Prob = Float64[], Rent = Float64[], ROI = Float64[])
    for (i,prob) in enumerate(probs)
        rent = prob * Spaces[i][2]
        roi = rent /Spaces[i][3] 
        push!(df,(Spaces[i][1], prob, rent,roi * 100))
    end
    df
end

summary()

#Monopoly simulaton with simple strategies

function simulate_game(n = 40, Spaces = Spaces)
    roi = summary()[!,:ROI]
    owned = zeros(Int,n)
    budget = 1500.0 * ones(3)
    position = ones(Int,3)
    seq = shuffle(1:3)
    while sum(budget.> 0.0) > 1
        for player in seq
            budget[player] ≤ 0.0 && continue
            roll = rand(2:12)
            position[player] =  mod(position[player] + roll - 1,n) + 1
            if position[player] == 31
                position[player] = 11
            elseif in(position[player],(3,18,34)) #community chest
                if rand() ≤ 2.0/16.0 
                    if rand() ≤ 0.5 #advance to go
                        position[player] = 1
                    else #go to jail
                        position[player] = 11
                    end
                end
            else
                owner = owned[position[player]] 
                (Spaces[position[player]][3] == Inf || owner == player) && continue
                if owner == 0
                    price = Spaces[position[player]][3]
                    price ≥ budget[player] && continue
                    #decision time
                    if player == 1  &&  rand() ≤ 0.5 
                        #random strategy (player 1)
                        owned[position[player]] = player
                        budget[player] -= price
                    elseif player == 2 
                        #always buy (player 2)
                        owned[position[player]] = player
                        budget[player] -= price
                    elseif player == 3 && roi[position[player]] > 0.15 
                        #buy only the best (player 3)
                        owned[position[player]] = player
                        budget[player] -= price
                    end  
                else
                    rent = Spaces[position[player]][2]
                    budget[player] -= rent
                    budget[owner] += rent
                end
                budget[player] ≤ 0.0 && replace!(owned, player => 0)
            end
        end
    end
    return budget
end

simulate_game()

n = 10000
res = zeros(3)
for k = 1:n
   res .+= (simulate_game() .> 0.0)
end
res ./ n
