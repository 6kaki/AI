def collate_fn(batch):
    return tuple(zip(*batch))

batch = [
    [1,2,3],
    [1,2,3],
    [1,2,3],
    [1,2,3]
]

print(list(zip(*batch)))