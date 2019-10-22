using Random, PyPlot

function schelling(n, p, sw)
    grid = [rand() < p ? rand([-1,1]) : 0 for i in 1:n, j in 1:n]
    t = 0
    img = imshow(grid)
    title("Step 0")
    moved = 1
    while !((moved == 0) || (t > n) && (moved > n^2/2))
        moved = 0
        agents = shuffle!([(x,y) for x in 1:n for y in 1:n if grid[x,y] != 0])
        for (x,y) in agents
            cs = 0
            ca = 0
            for dx in (-1, 0, 1), dy in (-1, 0, 1)
                if !(dx == 0 && dy == 0)
                    nx = 1 + mod(x+dx-1, n)
                    ny = 1 + mod(y+dy-1, n)
                    grid[nx,ny] != 0 && (ca += 1)
                    grid[nx,ny] == grid[x,y] && (cs += 1)
                end
            end
            if cs < sw * ca
                moved += 1
                while true
                    tx, ty = rand(1:n), rand(1:n)
                    if grid[tx, ty] == 0
                        grid[tx, ty] = grid[x, y]
                        grid[x, y] = 0
                        break
                    end
                end
            end
        end
        t += 1
        img[:set_data](grid)
        title("Step $t ($(round(100moved/n^2))%)")
        show()
        sleep(0.01)
    end
end

schelling(100, 0.75, 0.6)
