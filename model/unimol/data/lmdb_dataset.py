# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# HEART modification (see model/pretrain/IMPLEMENTATION_MAP.md, patch P1):
#   Upstream Uni-Mol enumerates every key at construction time
#   (`list(txn.cursor().iternext(values=False))`). For the 18.8M-record ligand
#   train.lmdb (~150 GB, network filesystem) that costs minutes of I/O and
#   >1 GB of RAM per process, while `__getitem__` never uses the key list: it
#   always reads key f"{idx}". We therefore take the length from the LMDB
#   statistics (O(1)) and verify that the numeric-key convention holds at the
#   first and last index. Set HEART_LMDB_FULL_KEY_SCAN=1 to restore the
#   upstream behaviour (useful for small fixtures or auditing).

import lmdb
import os
import pickle
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        try:
            with env.begin() as txn:
                if os.environ.get("HEART_LMDB_FULL_KEY_SCAN", "0") == "1":
                    keys = list(txn.cursor().iternext(values=False))
                    self._len = len(keys)
                    expected = {f"{i}".encode("ascii") for i in range(self._len)}
                    if set(keys) != expected:
                        raise ValueError(
                            "{}: keys are not the contiguous ASCII integers "
                            "0..{}".format(self.db_path, self._len - 1)
                        )
                else:
                    self._len = txn.stat()["entries"]
                    if self._len > 0:
                        for probe in (0, self._len - 1):
                            if txn.get(f"{probe}".encode("ascii")) is None:
                                raise ValueError(
                                    "{}: has {} entries but key b'{}' is missing; "
                                    "records must use contiguous ASCII integer keys "
                                    "0..N-1".format(self.db_path, self._len, probe)
                                )
        finally:
            # Never carry an open environment into forked DataLoader workers;
            # each worker reconnects lazily in __getitem__.
            env.close()
        logger.info("LMDBDataset {}: {} records".format(self.db_path, self._len))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return self._len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        if datapoint_pickled is None:
            raise KeyError("{}: key b'{}' not found".format(self.db_path, idx))
        data = pickle.loads(datapoint_pickled)
        return data
