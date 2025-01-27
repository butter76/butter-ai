import torch
import numpy as np


#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2021 Leela Chess Authors
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.


def parse_function(batch_tuple):
    """Convert raw binary data to tensors"""
    planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut = batch_tuple
    
    # Create writable copies using numpy first
    planes = np.frombuffer(planes, dtype=np.float32).copy()
    planes = torch.from_numpy(planes).reshape(-1, 112, 8, 8)
    
    probs = np.frombuffer(probs, dtype=np.float32).copy()
    probs = torch.from_numpy(probs).reshape(-1, 1858)
    
    winner = np.frombuffer(winner, dtype=np.float32).copy()
    winner = torch.from_numpy(winner).reshape(-1, 3)
    
    q = np.frombuffer(q, dtype=np.float32).copy()
    q = torch.from_numpy(q).reshape(-1, 3)
    
    plies_left = np.frombuffer(plies_left, dtype=np.float32).copy()
    plies_left = torch.from_numpy(plies_left).reshape(-1, 1)
    
    st_q = np.frombuffer(st_q, dtype=np.float32).copy()
    st_q = torch.from_numpy(st_q).reshape(-1, 3)
    
    opp_probs = np.frombuffer(opp_probs, dtype=np.float32).copy()
    opp_probs = torch.from_numpy(opp_probs).reshape(-1, 1858)
    
    next_probs = np.frombuffer(next_probs, dtype=np.float32).copy()
    next_probs = torch.from_numpy(next_probs).reshape(-1, 1858)
    
    fut = np.frombuffer(fut, dtype=np.float32).copy()
    fut = torch.from_numpy(fut).reshape(-1, 16, 12, 64)
    fut = fut.permute(0, 3, 1, 2)
    fut = torch.cat([fut, 1 - torch.sum(fut, dim=-1, keepdim=True)], dim=-1)
    
    return planes, probs, winner, q, plies_left, st_q, opp_probs, next_probs, fut