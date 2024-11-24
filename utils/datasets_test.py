from datasets import *


def check_MultiResolutionDataset(path="~/PatchInv/DATA/FlickrScenen"):  #
    """
    >>> check_MultiResolutionDataset()
     [*] Loaded data with length 50000
    45000, start from 0
    torch.Size([1, 3, 1024, 1024])
     [*] Loaded data with length 50000
    100, start from 45000
    torch.Size([1, 3, 1024, 1024])
    """
    dataset_test = MultiResolutionDataset(split=0.9, lmdb_root=path, train=True)
    print(f"{len(dataset_test)}, start from {dataset_test.st_pt}")
    loader_B = data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    for img in loader_B:
        print(img.shape)
        break
    dataset_test = MultiResolutionDataset(
        split=0.9, lmdb_root=path, train=False, test_num=100, resolution=(1024, 1024))
    print(f"{len(dataset_test)}, start from {dataset_test.st_pt}")
    loader_B = data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    for img in loader_B:
        print(img.shape)
        break


if __name__ == "__main__":
    import doctest
    doctest.testmod()
