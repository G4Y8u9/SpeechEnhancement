function randlist = getranditem(fullarray, N)
    assert(size(fullarray,1)>N);
    randindex = randperm(size(fullarray,1));
    randlist = fullarray(randindex(1:N),:);
end
