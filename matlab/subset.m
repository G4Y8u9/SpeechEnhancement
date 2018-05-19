function sub = subset(set, len)
    assert(length(set)>len);
    randrange = length(set)-len-1;
    randindex = randi(randrange);
    sub = set(randindex:randindex+len-1);
end
