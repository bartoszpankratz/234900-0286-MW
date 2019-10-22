using Statistics, PyPlot

function gendrift(n, p, doplot=true)
    flip = rand(n, n) .< p
    flop = Matrix{Bool}(undef, n, n)
    t = 0
    if doplot
        img = imshow(flip)
        title("Step 0 ($(mean(flip)))")
    end
    while count(x -> x != flip[1], flip) > 0
        for y in 1:n, x in 1:n
            neix = 1 + mod(x + rand(-2:0), n)
            neiy = 1 + mod(y + rand(-2:0), n)
            flop[x, y] = flip[neix, neiy]
        end
        t += 1
        flip, flop = flop, flip
        if doplot
            img[:set_data](flip)
            title("Step $t ($(mean(flip)))")
            show()
            sleep(0.01)
        end
    end
    (flip[1], t)
end

gendrift(20, 0.5)

