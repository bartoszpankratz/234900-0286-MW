function listoptions(grid, x, y)
    filter(p -> 1≤p[1]≤8 && 1≤p[2]≤8 && grid[p...] == 0,
           [(x+1, y+2), (x-1, y+2), (x+1, y-2), (x-1, y-2),
            (x+2, y+1), (x-2, y+1), (x+2, y-1), (x-2, y-1)])
end

function knight_jump!(grid=fill(0, 8, 8), (x,y)=(1,1), i=1)
    grid[x, y] = i
    i == 64 && return grid
    v = listoptions(grid, x, y)
    if !isempty(v)
        options = [length(listoptions(grid, p[1], p[2])) for p in v]
        v = v[sortperm(options)]
        for (nx, ny) in v
            if knight_jump!(grid, (nx, ny), i+1) !== nothing
                return grid
            end
        end
    end
    grid[x, y] = 0
    return nothing
end

knight_jump!()
