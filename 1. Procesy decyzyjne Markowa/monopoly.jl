
using DataFrames

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
        if !in(i,(3,18,30,33))
            S[i,i] = 1.0
        elseif i == 30
            S[i,10] = 1.0
        else #community chest
            #advance to go
            S[i,1] = 1.0 / 16.0;
            #go to jail
            S[i,10] = 1.0 / 16.0;
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

T = transition_matrix();

s = hcat(1.0,zeros(Float64,1,39));





