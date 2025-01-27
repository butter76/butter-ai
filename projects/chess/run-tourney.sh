#!/bin/bash

cutechess-cli -openings file=./book/8moves_v3.pgn format=pgn \
              -pgnout games.pgn \
              -engine cmd="./dist/random_bot" \
              -engine cmd="lc0" arg="--weights=/home/nreddy/lc0/models/CF-240M.pb.gz" depth=1 \
              -each tc=10/1 proto=uci \
              -games 2
