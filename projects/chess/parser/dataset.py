#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import yaml
import sys
import glob
import gzip
import random
import multiprocessing as mp
import itertools
from parser.chunkparsefunc import parse_function
from parser.chunkparser import ChunkParser
import random
import pickle
from torch.utils.data import Dataset, DataLoader


SKIP = 32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def fast_get_chunks(d):
    d = d.replace("*/", "")
    chunknames = []
    fo_chunknames = []
    subdirs = os.listdir(d)
    chunkfiles_name = "chunknames.pkl"
    if False and chunkfiles_name in subdirs: # TODO: remove False
        print(f"Using cached {d + chunkfiles_name}" )
        with open(d + chunkfiles_name, 'rb') as f:
            chunknames = pickle.load(f)
    else:

        i = 0
        for subdir in subdirs:
            if subdir.endswith(".gz"):
                fo_chunknames.append(d + subdir)
            else:
                prefix = d + subdir + "/"
                if os.path.isdir(prefix):
                    chunknames.append([prefix + s for s in os.listdir(prefix) if s.endswith(".gz")])

            i += 1
        chunknames.append(fo_chunknames)
            
        chunknames = list(itertools.chain.from_iterable(chunknames))

        with open(d + chunkfiles_name, 'wb') as f:
            print("Shuffling the chunks", flush=True)
            random.shuffle(chunknames)
            print(f"Caching {d + chunkfiles_name}" )
            pickle.dump(chunknames, f)

    return chunknames



def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_all_chunks(path, fast=False):

    if isinstance(path, list):
        print("getting chunks for", path)
        chunks = []
        for i in path:
            chunks += get_all_chunks(i, fast=fast)
        return chunks
    if fast:
        chunks = fast_get_chunks(path)
    else:
        chunks = []
        for d in glob.glob(path):
            chunks += get_chunks(d)
    print("got", len(chunks), "chunks for", path)
    return chunks


def get_latest_chunks(path, num_chunks, allow_less, sort_key_fn, fast=False):
    chunks = get_all_chunks(path, fast=fast)
    if len(chunks) < num_chunks:
        if allow_less:
            print("sorting {} chunks...".format(len(chunks)),
                  end="",
                  flush=True)
            if True:
                print("sorting disabled")
            else:
                chunks.sort(key=sort_key_fn, reverse=True)
            print("[done]")
            print("{} - {}".format(os.path.basename(chunks[-1]),
                                   os.path.basename(chunks[0])))
            print("shuffling chunks...", end="", flush=True)
            if True:
                print("shuffling disabled", flush=True)
            else:
                random.shuffle(chunks)
            print("[done]")
            return chunks
        else:
            print("Not enough chunks {}".format(len(chunks)))
            sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end="", flush=True)
    chunks.sort(key=sort_key_fn, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]),
                           os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


def identity_function(name):
    return name


def game_number_for_name(name):
    num_str = os.path.basename(name).upper().strip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ_-.")
    return int(num_str)


def get_input_mode(cfg):
    import parser.proto.net_pb2 as pb
    input_mode = cfg["model"].get("input_type", "classic")

    if input_mode == "classic":
        return pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    elif input_mode == "frc_castling":
        return pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
    elif input_mode == "canonical":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
    elif input_mode == "canonical_100":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
    elif input_mode == "canonical_armageddon":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
    elif input_mode == "canonical_v2":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
    elif input_mode == "canonical_v2_armageddon":
        return pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
    else:
        raise ValueError("Unknown input mode format: {}".format(input_mode))


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg["dataset"]["num_chunks"]
    allow_less = cfg["dataset"].get("allow_less_chunks", False)
    train_ratio = cfg["dataset"]["train_ratio"]
    fast_chunk_loading = cfg["dataset"].get("fast_chunk_loading", True)
    num_train = int(num_chunks * train_ratio)
    num_test = num_chunks - num_train
    sort_type = cfg["dataset"].get("sort_type", "mtime")
    if sort_type == "mtime":
        sort_key_fn = os.path.getmtime
    elif sort_type == "number":
        sort_key_fn = game_number_for_name
    elif sort_type == "name":
        sort_key_fn = identity_function
    else:
        raise ValueError("Unknown dataset sort_type: {}".format(sort_type))
    if "input_test" in cfg["dataset"]:
        train_chunks = get_latest_chunks(cfg["dataset"]["input_train"],
                                         num_train, allow_less, sort_key_fn, fast=fast_chunk_loading)
        test_chunks = get_latest_chunks(cfg["dataset"]["input_test"], num_test,
                                        allow_less, sort_key_fn, fast=fast_chunk_loading)
    else:
        chunks = get_latest_chunks(cfg["dataset"]["input"], num_chunks,
                                   allow_less, sort_key_fn, fast=fast_chunk_loading)
        if allow_less:
            num_train = int(len(chunks) * train_ratio)
            num_test = len(chunks) - num_train
        train_chunks = chunks[:num_train]
        test_chunks = chunks[num_train:]

    shuffle_size = cfg["training"]["shuffle_size"]
    total_batch_size = cfg["training"]["batch_size"]
    batch_splits = cfg["training"].get("num_batch_splits", 1)
    train_workers = cfg["dataset"].get("train_workers", None)
    test_workers = cfg["dataset"].get("test_workers", None)
    if total_batch_size % batch_splits != 0:
        raise ValueError("num_batch_splits must divide batch_size evenly")
    split_batch_size = total_batch_size // batch_splits

    diff_focus_min = cfg["training"].get("diff_focus_min", 1)
    diff_focus_slope = cfg["training"].get("diff_focus_slope", 0)
    diff_focus_q_weight = cfg["training"].get("diff_focus_q_weight", 6.0)
    diff_focus_pol_scale = cfg["training"].get("diff_focus_pol_scale", 3.5)
    pc_min = cfg["dataset"].get("pc_min")
    pc_max = cfg["dataset"].get("pc_max")

    root_dir = os.path.join(cfg["training"]["path"], cfg["name"])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    train_parser = ChunkParser(train_chunks,
                               get_input_mode(cfg),
                               shuffle_size=shuffle_size,
                               sample=SKIP,
                               batch_size=split_batch_size,
                               diff_focus_min=diff_focus_min,
                               diff_focus_slope=diff_focus_slope,
                               diff_focus_q_weight=diff_focus_q_weight,
                               diff_focus_pol_scale=diff_focus_pol_scale,
                               pc_min=pc_min,
                               pc_max=pc_max,
                               workers=train_workers)
    test_shuffle_size = int(shuffle_size * (1.0 - train_ratio))
    # no diff focus for test_parser
    test_parser = ChunkParser(test_chunks,
                              get_input_mode(cfg),
                              shuffle_size=test_shuffle_size,
                              sample=SKIP,
                              batch_size=split_batch_size,
                            #   pc_min=pc_min,
                            #   pc_max=pc_max,
                              workers=test_workers)
    
    
    if "input_validation" in cfg["dataset"]:
        valid_chunks = get_all_chunks(cfg["dataset"]["input_validation"], fast=fast_chunk_loading)
        validation_parser = ChunkParser(valid_chunks,
                                        get_input_mode(cfg),
                                        sample=1,
                                        batch_size=split_batch_size,
                                        # pc_min=pc_min,
                                        # pc_max=pc_max,
                                        workers=0)
        
    class ChessDataset(Dataset):
        def __init__(self, parser):
            self.parser = parser
            self.iterator = None
            
        def __iter__(self):
            return self.parser.parse()
            
        def __len__(self):
            # Since this is a streaming dataset, return a large number
            return 1000000
            
        def __getitem__(self, idx):
            if self.iterator is None:
                self.iterator = self.parser.parse()
            
            try:
                batch = next(self.iterator)
                # Process the batch using parse_function
                planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut = parse_function(batch)
                return planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut
                
            except StopIteration:
                # Reset iterator and try again
                self.iterator = self.parser.parse()
                batch = next(self.iterator)
                planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut = parse_function(batch)
                return planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut

    def create_dataloader(parser, batch_size, num_workers=0):
        dataset = ChessDataset(parser)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

    # # In main(), replace the TensorFlow dataset creation with:
    # train_dataloader = create_dataloader(
    #     train_parser,
    #     batch_size=split_batch_size,
    #     num_workers=train_workers if train_workers else 0
    # )

    # test_dataloader = create_dataloader(
    #     test_parser, 
    #     batch_size=split_batch_size,
    #     num_workers=test_workers if test_workers else 0
    # )

    # if 'input_validation' in cfg['dataset']:
    #     valid_dataloader = create_dataloader(
    #         validation_parser, # type: ignore
    #         batch_size=split_batch_size
    #     )

    for i in train_parser.parse():
        # Get first batch
        batch = parse_function(i)
        planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut = batch

        print("Planes tensor shape:", planes.shape)
        print("Probabilities tensor shape:", probs.shape) 
        print("Winner tensor shape:", winner.shape)
        print("Q tensor shape:", q.shape)
        print("Plies left tensor shape:", plies_left.shape)
        print("St Q tensor shape:", st_q.shape)
        print("Opponent probabilities tensor shape:", opp_probs.shape)
        print("Next probabilities tensor shape:", next_probs.shape)
        print("Future tensor shape:", fut.shape)

        # Print sample values
        print("\nSample values:")
        print("First plane:\n", planes[0,0,:8,:8])
        print("First probability distribution:\n", probs[0,:10])
        print("First winner values:", winner[0])
        print("First Q values:", q[0])
        print("First plies left:", plies_left[0])
        print("First st Q values:", st_q[0])
        print("First opponent probabilities:", opp_probs[0,:10])
        print("First next probabilities:", next_probs[0,:10])
        print("First future values:", fut[0,0,:5,:5])
        break
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg',
                           type=argparse.FileType('r'),
                           help='yaml configuration with training parameters')
    argparser.add_argument('--output',
                           type=str,
                           help='file to store weights in')

    #mp.set_start_method('spawn')
    main(argparser.parse_args())
    mp.freeze_support()