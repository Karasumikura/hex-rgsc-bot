import heapq
import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None


BOARD_SIZE = 11
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
BLACK = 1
WHITE = 2
EMPTY = 0
# Official Hex matches do not inject a fixed first move.  The competition CLI
# below lets the network choose the first move; the legacy adapter only keeps
# a generic emergency coordinate for malformed/full-board input.
EMERGENCY_MOVE_BOT = (0, 0)
NEIGHBORS = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
MODEL_BASENAMES = ("weight_iter_78000.pkl", "weight_iter_78000.pt")

KEEP_RUNNING_SENTINEL = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
# Botzone long-running mode must be explicitly requested after each move.
# Default to enabled for the submission build; local one-shot testing can set
# BOTZONE_KEEP_RUNNING=0 to restore per-turn exit behavior.
ENABLE_KEEP_RUNNING = os.environ.get("BOTZONE_KEEP_RUNNING", "1") != "0"

# Legacy Botzone per-move limits.  The official CLI uses the 30-minute clock
# below instead and passes a dynamically allocated budget to MCTS.
# Botzone's per-turn limit is kept for the legacy JSON adapter.  The official
# competition CLI uses the 30-minute clock below instead.
FIRST_MOVE_TIME_BUDGET = 7.7
NORMAL_TIME_BUDGET = 3.8
SAFETY_MARGIN = 0.15
MAX_SIMULATIONS_NORMAL = 90000
MAX_SIMULATIONS_DANGER = 90000

COMPETITION_TOTAL_TIME_SEC = 30.0 * 60.0
COMPETITION_TIME_RESERVE_SEC = 180.0
COMPETITION_OPENING_CAP_SEC = 25.0
COMPETITION_NORMAL_CAP_SEC = 45.0
COMPETITION_DANGER_CAP_SEC = 90.0
COMPETITION_DANGER_MULTIPLIER = 1.75
COMPETITION_FAST_THRESHOLD_SEC = 180.0
COMPETITION_FAST_CAP_SEC = 5.0
COMPETITION_HARD_STOP_MARGIN_SEC = 0.25
COMPETITION_MIN_MOVE_BUDGET_SEC = 0.30

C_PUCT = 1.25
ROOT_TOPK_EXPAND = 32
CHILD_TOPK_EXPAND = 16
MIN_POLICY_PRIOR = 1e-8
DANGER_PRIOR_BOOST = 3.0
BRIDGE_PRIOR_BOOST = 1.75
DOUBLE_THREAT_PRIOR_PENALTY = 0.08
TIME_CHECK_INTERVAL = 8
TACTICAL_SCAN_LEGAL_THRESHOLD = 24
EVAL_CACHE_LIMIT = 4096


if torch is not None:
    class ResidualBlock(nn.Module):
        def __init__(self, num_channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(num_channels)

        def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(residual + x)


    class PolicyNetwork(nn.Module):
        def __init__(self, num_channels, channel_height, channel_width, action_size):
            super(PolicyNetwork, self).__init__()
            num_output_channels = int(math.ceil(float(action_size) / float(channel_height * channel_width)))
            self.channel_height = channel_height
            self.channel_width = channel_width
            self.num_output_channels = num_output_channels
            self.conv = nn.Conv2d(num_channels, num_output_channels, kernel_size=1)
            self.bn = nn.BatchNorm2d(num_output_channels)
            self.fc = nn.Linear(num_output_channels * channel_height * channel_width, action_size)

        def forward(self, x):
            x = F.relu(self.bn(self.conv(x)))
            x = x.view(-1, self.num_output_channels * self.channel_height * self.channel_width)
            return self.fc(x)


    class ValueNetwork(nn.Module):
        def __init__(self, num_channels, channel_height, channel_width, hidden_channels):
            super(ValueNetwork, self).__init__()
            self.channel_height = channel_height
            self.channel_width = channel_width
            self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
            self.bn = nn.BatchNorm2d(1)
            self.fc1 = nn.Linear(channel_height * channel_width, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, 1)

        def forward(self, x):
            x = F.relu(self.bn(self.conv(x)))
            x = x.view(-1, self.channel_height * self.channel_width)
            x = F.relu(self.fc1(x))
            return torch.tanh(self.fc2(x))


    class RegretNetwork(nn.Module):
        def __init__(self, num_channels, channel_height, channel_width, hidden_channels):
            super(RegretNetwork, self).__init__()
            self.channel_height = channel_height
            self.channel_width = channel_width
            self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
            self.bn = nn.BatchNorm2d(1)
            self.fc1 = nn.Linear(channel_height * channel_width, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, 1)

        def forward(self, x):
            x = F.relu(self.bn(self.conv(x)))
            x = x.view(-1, self.channel_height * self.channel_width)
            x = F.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x)) * 4.0


    class RankingNetwork(nn.Module):
        def __init__(self, num_channels, channel_height, channel_width, hidden_channels):
            super(RankingNetwork, self).__init__()
            self.channel_height = channel_height
            self.channel_width = channel_width
            self.conv = nn.Conv2d(num_channels, 1, kernel_size=1)
            self.bn = nn.BatchNorm2d(1)
            self.fc1 = nn.Linear(channel_height * channel_width, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, 1)

        def forward(self, x):
            x = F.relu(self.bn(self.conv(x)))
            x = x.view(-1, self.channel_height * self.channel_width)
            x = F.relu(self.fc1(x))
            return self.fc2(x)


    class AlphaZeroNetwork(nn.Module):
        def __init__(self):
            super(AlphaZeroNetwork, self).__init__()
            self.conv = nn.Conv2d(4, 256, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(256)
            self.residual_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(3)])
            self.policy = PolicyNetwork(256, BOARD_SIZE, BOARD_SIZE, ACTION_SIZE)
            self.value = ValueNetwork(256, BOARD_SIZE, BOARD_SIZE, 256)
            self.regret = RegretNetwork(256, BOARD_SIZE, BOARD_SIZE, 256)
            self.ranking = RankingNetwork(256, BOARD_SIZE, BOARD_SIZE, 256)

        def forward(self, state):
            x = F.relu(self.bn(self.conv(state)))
            for block in self.residual_blocks:
                x = block(x)
            policy_logit = self.policy(x)
            policy = torch.softmax(policy_logit, dim=1)
            value = self.value(x)
            opening_value = self.regret(x.detach())
            opening_ranking = self.ranking(x.detach())
            return {
                "policy_logit": policy_logit,
                "policy": policy,
                "value": value,
                "opening_value": opening_value,
                "opening_ranking": opening_ranking,
            }


def normalize_record(record):
    if isinstance(record, dict):
        return record
    if isinstance(record, str):
        try:
            parsed = json.loads(record)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def bot_to_internal(x, y):
    return y, BOARD_SIZE - 1 - x


def internal_to_bot(row, col):
    return BOARD_SIZE - 1 - col, row


def inside(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def opponent_of(stone):
    return WHITE if stone == BLACK else BLACK


def empty_board():
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def copy_board(board):
    return [row[:] for row in board]


def board_key(board):
    return tuple(cell for row in board for cell in row)


def apply_internal_move(board, move, stone):
    board[move[0]][move[1]] = stone


def undo_internal_move(board, move):
    board[move[0]][move[1]] = EMPTY


def place_bot_move(board, move, stone):
    x, y = move
    if x < 0 or y < 0:
        return
    row, col = bot_to_internal(x, y)
    if inside(row, col):
        board[row][col] = stone


def parse_move(record):
    obj = normalize_record(record)
    if "x" not in obj or "y" not in obj:
        return None
    try:
        x = int(obj["x"])
        y = int(obj["y"])
    except Exception:
        return None
    if x < 0 or y < 0:
        return None
    if x >= BOARD_SIZE or y >= BOARD_SIZE:
        return None
    return x, y


def is_full_input(obj):
    return isinstance(obj, dict) and isinstance(obj.get("requests"), list) and isinstance(obj.get("responses"), list)


def extract_forced(record):
    obj = normalize_record(record)
    if "forced_x" not in obj or "forced_y" not in obj:
        return None
    try:
        return int(obj["forced_x"]), int(obj["forced_y"])
    except Exception:
        return None


def count_ply(board):
    total = 0
    for row in board:
        for cell in row:
            if cell != EMPTY:
                total += 1
    return total


def reconstruct_position(full_input):
    requests = full_input.get("requests", [])
    responses = full_input.get("responses", [])
    board = empty_board()

    first_request = normalize_record(requests[0]) if requests else {}
    we_are_black = int(first_request.get("x", -1)) < 0
    our_stone = BLACK if we_are_black else WHITE
    opp_stone = opponent_of(our_stone)

    if our_stone == BLACK:
        for idx, response in enumerate(responses):
            move = parse_move(response)
            if move is not None:
                place_bot_move(board, move, BLACK)
            if idx + 1 < len(requests):
                move = parse_move(requests[idx + 1])
                if move is not None:
                    place_bot_move(board, move, WHITE)
    else:
        for idx, response in enumerate(responses):
            if idx < len(requests):
                move = parse_move(requests[idx])
                if move is not None:
                    place_bot_move(board, move, BLACK)
            move = parse_move(response)
            if move is not None:
                place_bot_move(board, move, WHITE)
        if len(requests) > len(responses):
            move = parse_move(requests[-1])
            if move is not None:
                place_bot_move(board, move, BLACK)

    current_request = normalize_record(requests[-1]) if requests else {}
    forced = extract_forced(current_request)
    last_opp_move = parse_move(current_request)
    ply = count_ply(board)

    return {
        "board": board,
        "our_stone": our_stone,
        "opp_stone": opp_stone,
        "forced": forced,
        "ply": ply,
        "requests": requests,
        "responses": responses,
        "last_opp_move": last_opp_move,
    }


def get_legal_moves(board):
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                moves.append((row, col))
    return moves


def action_id(move):
    return move[0] * BOARD_SIZE + move[1]


def build_features(board, to_play):
    opp = opponent_of(to_play)
    features = [[[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)] for _ in range(4)]
    for row in range(BOARD_SIZE):
        own_plane = features[0][row]
        opp_plane = features[1][row]
        for col in range(BOARD_SIZE):
            stone = board[row][col]
            if stone == to_play:
                own_plane[col] = 1.0
            elif stone == opp:
                opp_plane[col] = 1.0
    turn_value = 1.0 if to_play == BLACK else 0.0
    opp_turn_value = 1.0 - turn_value
    for row in range(BOARD_SIZE):
        features[2][row] = [turn_value] * BOARD_SIZE
        features[3][row] = [opp_turn_value] * BOARD_SIZE
    return features


def has_winning_path(board, stone):
    visited = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    stack = []

    if stone == BLACK:
        for row in range(BOARD_SIZE):
            if board[row][0] == BLACK:
                stack.append((row, 0))
                visited[row][0] = True
        while stack:
            row, col = stack.pop()
            if col == BOARD_SIZE - 1:
                return True
            for dr, dc in NEIGHBORS:
                nr = row + dr
                nc = col + dc
                if inside(nr, nc) and not visited[nr][nc] and board[nr][nc] == BLACK:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
    else:
        for col in range(BOARD_SIZE):
            if board[0][col] == WHITE:
                stack.append((0, col))
                visited[0][col] = True
        while stack:
            row, col = stack.pop()
            if row == BOARD_SIZE - 1:
                return True
            for dr, dc in NEIGHBORS:
                nr = row + dr
                nc = col + dc
                if inside(nr, nc) and not visited[nr][nc] and board[nr][nc] == WHITE:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
    return False


def immediate_winning_moves(board, stone, legal_moves):
    wins = []
    for row, col in legal_moves:
        board[row][col] = stone
        if has_winning_path(board, stone):
            wins.append((row, col))
        board[row][col] = EMPTY
    return wins


def count_immediate_wins(board, stone, limit):
    count = 0
    for row in range(BOARD_SIZE):
        board_row = board[row]
        for col in range(BOARD_SIZE):
            if board_row[col] != EMPTY:
                continue
            board_row[col] = stone
            if has_winning_path(board, stone):
                count += 1
                board_row[col] = EMPTY
                if count >= limit:
                    return count
            else:
                board_row[col] = EMPTY
    return count


def local_shape_score(board, move, stone):
    row, col = move
    opp = opponent_of(stone)
    own_adj = 0
    opp_adj = 0
    for dr, dc in NEIGHBORS:
        nr = row + dr
        nc = col + dc
        if not inside(nr, nc):
            continue
        if board[nr][nc] == stone:
            own_adj += 1
        elif board[nr][nc] == opp:
            opp_adj += 1
    center_bias = 5.0 - 0.35 * (abs(row - 5) + abs(col - 5))
    axis = col if stone == BLACK else row
    side_bias = 0.2 * min(axis, BOARD_SIZE - 1 - axis)
    return own_adj * 0.8 + opp_adj * 1.0 + center_bias + side_bias


def min_connection_cost(board, stone):
    inf = 10 ** 9
    dist = [[inf for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    heap = []

    def cell_cost(value):
        if value == stone:
            return 0
        if value == EMPTY:
            return 1
        return inf

    if stone == BLACK:
        for row in range(BOARD_SIZE):
            cost = cell_cost(board[row][0])
            if cost < inf:
                dist[row][0] = cost
                heapq.heappush(heap, (cost, row, 0))
    else:
        for col in range(BOARD_SIZE):
            cost = cell_cost(board[0][col])
            if cost < inf:
                dist[0][col] = cost
                heapq.heappush(heap, (cost, 0, col))

    while heap:
        cur_dist, row, col = heapq.heappop(heap)
        if cur_dist != dist[row][col]:
            continue
        if stone == BLACK and col == BOARD_SIZE - 1:
            return cur_dist
        if stone == WHITE and row == BOARD_SIZE - 1:
            return cur_dist
        for dr, dc in NEIGHBORS:
            nr = row + dr
            nc = col + dc
            if not inside(nr, nc):
                continue
            step = cell_cost(board[nr][nc])
            if step >= inf:
                continue
            nd = cur_dist + step
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return BOARD_SIZE * BOARD_SIZE


def best_center_tiebreak(moves, policy=None):
    best = None
    best_key = None
    for move in moves:
        row, col = move
        prob = 0.0 if policy is None else float(policy[action_id(move)])
        center = -(abs(row - 5) + abs(col - 5))
        key = (prob, center, -row, -col)
        if best_key is None or key > best_key:
            best_key = key
            best = move
    return best


def bridge_response_moves(board, our_stone, opp_last_move):
    # Tactical bridge-save hook intentionally kept lightweight.
    # The main playing strength now comes from neural PUCT-MCTS.
    return []


if torch is not None:
    class RGSCModel(object):
        def __init__(self):
            self.model = None
            self.kind = "heuristic"
            self.path = ""
            self.error = ""
            self.allow_torchscript = os.environ.get("RGSC_ALLOW_TORCHSCRIPT", "") == "1"
            self.eval_cache = {}

            try:
                torch.set_grad_enabled(False)
                if hasattr(torch, "set_num_threads"):
                    torch.set_num_threads(1)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(1)
            except Exception:
                pass

            for path in self._candidate_paths("weight_iter_78000.pkl"):
                if self._try_load_state_dict(path):
                    return
            if self.allow_torchscript:
                for path in self._candidate_paths("weight_iter_78000.pt"):
                    if self._try_load_torchscript(path):
                        return
            if not self.error:
                self.error = "no model file found"

        def _candidate_paths(self, basename):
            seen = set()
            roots = [
                os.getcwd(),
                os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(),
            ]
            module_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
            roots.append(os.path.dirname(module_dir))
            paths = []
            for root in roots:
                for rel in (
                    os.path.join("data", basename),
                    os.path.join("data", "models", basename),
                    os.path.join("models", basename),
                    basename,
                ):
                    full = os.path.abspath(os.path.join(root, rel))
                    if full not in seen:
                        seen.add(full)
                        paths.append(full)
            return paths

        def _finalize_model(self, model, kind, path):
            previous = (self.model, self.kind, self.path, self.error)
            try:
                model.eval()
                # Run one real forward pass before publishing the model as ready.
                self.model = model
                self.eval_cache = {}
                self.evaluate_position(empty_board(), BLACK)
            except Exception:
                self.model, self.kind, self.path, self.error = previous
                self.eval_cache = {}
                raise
            self.kind = kind
            self.path = path
            self.error = ""
            return True

        def _try_load_torchscript(self, path):
            if not os.path.exists(path):
                return False
            try:
                model = torch.jit.load(path, map_location="cpu")
                return self._finalize_model(model, "torchscript", path)
            except Exception as exc:
                self.error = "torchscript: {0}".format(exc)
                return False

        def _try_load_state_dict(self, path):
            if not os.path.exists(path):
                return False
            try:
                snapshot = torch.load(path, map_location="cpu")
                state_dict = snapshot["network"] if isinstance(snapshot, dict) and "network" in snapshot else snapshot
                model = AlphaZeroNetwork()
                model.load_state_dict(state_dict)
                return self._finalize_model(model, "state_dict", path)
            except Exception as exc:
                self.error = "state_dict: {0}".format(exc)
                return False

        @property
        def ready(self):
            return self.model is not None

        def evaluate_batch(self, feature_batch):
            if not self.ready or not feature_batch:
                return []
            tensor = torch.tensor(feature_batch, dtype=torch.float32)
            with torch.no_grad():
                output = self.model(tensor)
            policy = output["policy"].cpu()
            value = output["value"].cpu()

            results = []
            for idx in range(policy.size(0)):
                results.append({
                    "policy": policy[idx].tolist(),
                    "value": float(value[idx].view(-1)[0]),
                })
            return results

        def evaluate_position(self, board, to_play):
            if not self.ready:
                return None, 0.0
            key = (to_play, board_key(board))
            if key in self.eval_cache:
                return self.eval_cache[key]

            result = self.evaluate_batch([build_features(board, to_play)])[0]
            policy = result["policy"]
            value = float(result["value"])

            if len(self.eval_cache) >= EVAL_CACHE_LIMIT:
                self.eval_cache.clear()
            self.eval_cache[key] = (policy, value)
            return policy, value
else:
    class RGSCModel(object):
        def __init__(self):
            self.model = None
            self.kind = "heuristic"
            self.path = ""
            self.error = "torch import failed"

        @property
        def ready(self):
            return False

        def evaluate_batch(self, feature_batch):
            return []

        def evaluate_position(self, board, to_play):
            return None, 0.0


class MCTSNode(object):
    def __init__(self, parent, move, to_play, prior):
        self.parent = parent
        self.move = move
        self.to_play = to_play
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False
        self.terminal_value = None

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / float(self.visit_count)

    def expanded(self):
        return self.is_expanded or self.terminal_value is not None

    def select_child(self, c_puct):
        if not self.children:
            return None
        sqrt_parent = math.sqrt(float(max(1, self.visit_count)))
        best_child = None
        best_key = None
        for child in self.children.values():
            q = -child.q_value()
            u = c_puct * child.prior * sqrt_parent / float(1 + child.visit_count)
            center = -(abs(child.move[0] - 5) + abs(child.move[1] - 5))
            key = (q + u, q, child.prior, center)
            if best_key is None or key > best_key:
                best_key = key
                best_child = child
        return best_child

    def expand(self, move_priors):
        self.children = {}
        next_to_play = opponent_of(self.to_play)
        for move, prior in move_priors:
            self.children[move] = MCTSNode(self, move, next_to_play, prior)
        self.is_expanded = True

    def backup(self, value):
        node = self
        cur_value = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += cur_value
            cur_value = -cur_value
            node = node.parent


class NeuralMCTS(object):
    def __init__(self, model):
        self.model = model
        self.root_value = 0.0
        self.root_expanded = 0
        self.danger_info = None
        self.root_restrict_moves = None
        self.bridge_moves = set()

    def _policy_prior(self, policy, move):
        if policy is None:
            return MIN_POLICY_PRIOR
        return max(MIN_POLICY_PRIOR, float(policy[action_id(move)]))

    def analyze_root_danger(self, board, to_play, legal_moves):
        opp = opponent_of(to_play)
        my_path = min_connection_cost(board, to_play)
        opp_path = min_connection_cost(board, opp)
        danger = opp_path <= 4
        candidates = set()
        if danger:
            for move in legal_moves:
                apply_internal_move(board, move, to_play)
                opp_after = min_connection_cost(board, opp)
                if opp_after > opp_path:
                    candidates.add(move)
                undo_internal_move(board, move)
        return {
            "danger": danger,
            "my_path": my_path,
            "opp_path": opp_path,
            "candidates": candidates,
        }

    def _select_candidate_moves(self, board, to_play, policy, is_root):
        legal_moves = self.root_restrict_moves if (is_root and self.root_restrict_moves) else get_legal_moves(board)
        legal_moves = [move for move in legal_moves if board[move[0]][move[1]] == EMPTY]
        if not legal_moves:
            return [], None

        candidate_pool = legal_moves
        leaf_value_override = None
        run_tactical_scan = is_root or len(legal_moves) <= TACTICAL_SCAN_LEGAL_THRESHOLD

        own_wins = []
        opp_wins = []
        if run_tactical_scan:
            own_wins = immediate_winning_moves(board, to_play, legal_moves)
            if own_wins:
                candidate_pool = own_wins
                leaf_value_override = 1.0
            else:
                opp_wins = immediate_winning_moves(board, opponent_of(to_play), legal_moves)
                if len(opp_wins) > 1:
                    candidate_pool = opp_wins
                    if not is_root:
                        leaf_value_override = -1.0
                elif len(opp_wins) == 1:
                    candidate_pool = opp_wins

        force_include = set()
        if is_root and self.danger_info and self.danger_info["danger"]:
            force_include.update(self.danger_info["candidates"])
        if is_root and self.bridge_moves:
            force_include.update(self.bridge_moves)

        topk = ROOT_TOPK_EXPAND if is_root else CHILD_TOPK_EXPAND
        scored = []
        for move in candidate_pool:
            prior = self._policy_prior(policy, move)
            boost = 1.0
            if is_root and self.danger_info and move in self.danger_info["candidates"]:
                boost *= DANGER_PRIOR_BOOST
            if is_root and move in self.bridge_moves:
                boost *= BRIDGE_PRIOR_BOOST
            center = -(abs(move[0] - 5) + abs(move[1] - 5))
            shape = local_shape_score(board, move, to_play)
            scored.append((prior * boost, prior, shape, center, move))
        scored.sort(reverse=True)

        selected = []
        selected_set = set()
        keep_all = bool(own_wins) or len(candidate_pool) <= topk or len(opp_wins) >= 1
        if keep_all:
            for _, _, _, _, move in scored:
                selected.append(move)
                selected_set.add(move)
        else:
            forced_rank = []
            for move in force_include:
                if move in candidate_pool:
                    forced_rank.append((self._policy_prior(policy, move), move))
            forced_rank.sort(reverse=True)
            for _, move in forced_rank:
                if move not in selected_set:
                    selected.append(move)
                    selected_set.add(move)
            for _, _, _, _, move in scored:
                if move in selected_set:
                    continue
                selected.append(move)
                selected_set.add(move)
                if len(selected) >= topk:
                    break

        bad_moves = set()
        if is_root and selected:
            opp = opponent_of(to_play)
            for move in selected:
                apply_internal_move(board, move, to_play)
                if count_immediate_wins(board, opp, 2) >= 2:
                    bad_moves.add(move)
                undo_internal_move(board, move)

        move_priors = []
        prior_sum = 0.0
        for move in selected:
            prior = self._policy_prior(policy, move)
            if is_root and self.danger_info and move in self.danger_info["candidates"]:
                prior *= DANGER_PRIOR_BOOST
            if is_root and move in self.bridge_moves:
                prior *= BRIDGE_PRIOR_BOOST
            if move in bad_moves:
                prior *= DOUBLE_THREAT_PRIOR_PENALTY
            prior = max(MIN_POLICY_PRIOR, prior)
            move_priors.append([move, prior])
            prior_sum += prior

        if prior_sum <= 0.0:
            uniform = 1.0 / float(max(1, len(move_priors)))
            return [(item[0], uniform) for item in move_priors], leaf_value_override

        return [(item[0], item[1] / prior_sum) for item in move_priors], leaf_value_override

    def _simulate(self, root, board):
        node = root
        while True:
            if node.terminal_value is not None:
                leaf_value = node.terminal_value
                break

            if not node.is_expanded:
                policy, value = self.model.evaluate_position(board, node.to_play)
                is_root = node.parent is None
                move_priors, leaf_value_override = self._select_candidate_moves(board, node.to_play, policy, is_root)
                if is_root:
                    self.root_value = value
                    self.root_expanded = len(move_priors)
                if leaf_value_override is not None:
                    value = leaf_value_override
                if move_priors:
                    node.expand(move_priors)
                else:
                    node.terminal_value = 0.0
                    value = 0.0
                leaf_value = value
                break

            child = node.select_child(C_PUCT)
            if child is None:
                leaf_value = 0.0
                break

            apply_internal_move(board, child.move, node.to_play)
            node = child
            prev_player = opponent_of(node.to_play)
            if has_winning_path(board, prev_player):
                node.terminal_value = -1.0
                leaf_value = node.terminal_value
                break

        node.backup(leaf_value)

    def _best_final_child(self, root):
        best_child = None
        best_key = None
        for child in root.children.values():
            q = -child.q_value()
            center = -(abs(child.move[0] - 5) + abs(child.move[1] - 5))
            key = (child.visit_count, q, child.prior, center)
            if best_key is None or key > best_key:
                best_key = key
                best_child = child
        return best_child

    def search(self, board, to_play, time_budget_sec, max_simulations, root_restrict_moves=None, bridge_moves=None):
        legal_moves = get_legal_moves(board)
        self.root_restrict_moves = list(root_restrict_moves) if root_restrict_moves else None
        self.bridge_moves = set(bridge_moves or [])
        self.danger_info = self.analyze_root_danger(board, to_play, legal_moves)

        root = MCTSNode(None, None, to_play, 1.0)
        deadline = time.perf_counter() + max(0.05, time_budget_sec - SAFETY_MARGIN)
        sims = 0

        while sims < max_simulations:
            if sims > 0 and sims % TIME_CHECK_INTERVAL == 0 and time.perf_counter() >= deadline:
                break
            self._simulate(root, copy_board(board))
            sims += 1

        if not root.children:
            move = best_center_tiebreak(legal_moves)
            return move, {
                "mode": "puct-mcts",
                "sims": sims,
                "root_v": round(self.root_value, 4),
                "best_n": 0,
                "best_q": 0.0,
                "best_p": 0.0,
                "legal": len(legal_moves),
                "expanded": 0,
                "danger": self.danger_info["danger"],
                "my_path": self.danger_info["my_path"],
                "opp_path": self.danger_info["opp_path"],
                "danger_candidates": len(self.danger_info["candidates"]),
            }

        best_child = self._best_final_child(root)
        best_q = -best_child.q_value()
        return best_child.move, {
            "mode": "puct-mcts",
            "sims": sims,
            "root_v": round(self.root_value, 4),
            "best_n": best_child.visit_count,
            "best_q": round(best_q, 4),
            "best_p": round(best_child.prior, 4),
            "legal": len(legal_moves),
            "expanded": self.root_expanded,
            "danger": self.danger_info["danger"],
            "my_path": self.danger_info["my_path"],
            "opp_path": self.danger_info["opp_path"],
            "danger_candidates": len(self.danger_info["candidates"]),
        }


def run_mcts(model, board, our_stone, opp_stone, time_budget_sec, max_simulations, root_restrict_moves=None, bridge_moves=None):
    del opp_stone
    mcts = NeuralMCTS(model)
    return mcts.search(board, our_stone, time_budget_sec, max_simulations, root_restrict_moves, bridge_moves)


def choose_heuristic_move(board, our_stone, opp_stone, candidate_moves):
    best_move = None
    best_key = None
    best_debug = None

    for move in candidate_moves:
        apply_internal_move(board, move, our_stone)
        opp_immediate = count_immediate_wins(board, opp_stone, 2)
        my_path = min_connection_cost(board, our_stone)
        opp_path = min_connection_cost(board, opp_stone)
        shape = local_shape_score(board, move, our_stone)
        undo_internal_move(board, move)

        score = 0.12 * (opp_path - my_path)
        score += 0.08 * shape
        score -= 0.9 * opp_immediate
        center = -(abs(move[0] - 5) + abs(move[1] - 5))
        key = (score, center)
        if best_key is None or key > best_key:
            best_key = key
            best_move = move
            best_debug = {
                "mode": "heuristic-fallback",
                "score": round(score, 4),
                "opp_immediate": opp_immediate,
                "my_path": my_path,
                "opp_path": opp_path,
            }

    return best_move, best_debug


class HexBot(object):
    def __init__(self):
        self.model = RGSCModel()
        self.board = empty_board()
        self.our_stone = None
        self.opp_stone = None
        self.forced = None
        self.last_opp_move = None
        self.initialized = False

    def _load_from_full_input(self, full_input):
        position = reconstruct_position(full_input)
        self.board = copy_board(position["board"])
        self.our_stone = position["our_stone"]
        self.opp_stone = position["opp_stone"]
        self.forced = position["forced"]
        self.last_opp_move = position["last_opp_move"]
        self.initialized = True

    def _apply_incremental_request(self, request):
        if not self.initialized:
            raise ValueError("incremental request received before initialization")
        move = parse_move(request)
        if move is not None:
            place_bot_move(self.board, move, self.opp_stone)
            self.last_opp_move = move
        forced = extract_forced(request)
        if forced is not None:
            self.forced = forced

    def _commit_our_move(self, bot_move):
        if self.initialized:
            place_bot_move(self.board, bot_move, self.our_stone)

    def _commit_internal_move(self, move):
        if self.initialized and move is not None:
            apply_internal_move(self.board, move, self.our_stone)

    def _base_debug(self):
        debug = {"model": self.model.kind}
        if self.model.path:
            debug["path"] = os.path.basename(self.model.path)
        if self.model.error and not self.model.ready:
            debug["model_error"] = self.model.error[:120]
        return debug

    def choose_internal_move(self, time_budget_sec=None):
        """Choose and commit one move in the engine's internal coordinates.

        The competition CLI uses this method directly so it can expose the
        official Hex notation.  The legacy Botzone adapter converts the same
        internal move back to Botzone's x/y coordinates.
        """
        if self.our_stone is None or self.opp_stone is None:
            raise ValueError("bot side is not initialized")
        board = self.board
        our_stone = self.our_stone
        opp_stone = self.opp_stone
        ply = count_ply(board)

        legal_moves = get_legal_moves(board)
        if not legal_moves:
            debug = self._base_debug()
            debug["mode"] = "empty-fallback"
            return None, debug

        winning_moves = immediate_winning_moves(board, our_stone, legal_moves)
        if winning_moves:
            if self.model.ready:
                policy, root_value = self.model.evaluate_position(board, our_stone)
            else:
                policy, root_value = None, 0.0
            move = best_center_tiebreak(winning_moves, policy)
            self._commit_internal_move(move)
            debug = self._base_debug()
            debug.update({
                "mode": "immediate-win",
                "root_v": round(root_value, 4),
                "legal": len(legal_moves),
                "choices": len(winning_moves),
            })
            return move, debug

        must_block = immediate_winning_moves(board, opp_stone, legal_moves)
        if len(must_block) == 1:
            move = must_block[0]
            self._commit_internal_move(move)
            debug = self._base_debug()
            debug.update({
                "mode": "forced-block",
                "legal": len(legal_moves),
                "must_block": 1,
            })
            return move, debug

        bridge_moves = bridge_response_moves(board, our_stone, self.last_opp_move)
        if len(bridge_moves) == 1:
            move = bridge_moves[0]
            self._commit_internal_move(move)
            debug = self._base_debug()
            debug.update({
                "mode": "bridge-response",
                "legal": len(legal_moves),
            })
            return move, debug

        if self.model.ready:
            first_decision = ply <= 1
            time_budget = time_budget_sec
            if time_budget is None:
                time_budget = FIRST_MOVE_TIME_BUDGET if first_decision else NORMAL_TIME_BUDGET
            opp_path_before = min_connection_cost(board, opp_stone)
            danger_mode = opp_path_before <= 4 or len(must_block) > 1
            max_simulations = MAX_SIMULATIONS_DANGER if danger_mode else MAX_SIMULATIONS_NORMAL
            root_restrict_moves = must_block if len(must_block) > 1 else None

            move, debug = run_mcts(
                self.model,
                board,
                our_stone,
                opp_stone,
                time_budget,
                max_simulations,
                root_restrict_moves=root_restrict_moves,
                bridge_moves=bridge_moves,
            )
            self._commit_internal_move(move)
            debug.update(self._base_debug())
            if len(must_block) > 1:
                debug["mode"] = "multi-block-mcts"
                debug["must_block"] = len(must_block)
            else:
                debug["must_block"] = 0
            return move, debug

        move, debug = choose_heuristic_move(board, our_stone, opp_stone, legal_moves)
        self._commit_internal_move(move)
        debug.update(self._base_debug())
        debug["legal"] = len(legal_moves)
        debug["must_block"] = len(must_block)
        return move, debug

    def competition_time_budget(self, elapsed_seconds, danger=False):
        """Return a per-move budget that preserves the full match clock."""
        if not self.initialized:
            raise ValueError("bot side is not initialized")

        legal_count = len(get_legal_moves(self.board))
        remaining_our_moves = max(1, (legal_count + 1) // 2)
        remaining_seconds = max(0.05, COMPETITION_TOTAL_TIME_SEC - elapsed_seconds)
        if remaining_seconds <= COMPETITION_HARD_STOP_MARGIN_SEC:
            return max(0.05, remaining_seconds * 0.5)

        spendable = max(0.0, remaining_seconds - COMPETITION_TIME_RESERVE_SEC)
        fair_share = spendable / float(remaining_our_moves)

        if remaining_seconds <= COMPETITION_FAST_THRESHOLD_SEC:
            cap = COMPETITION_FAST_CAP_SEC
        elif count_ply(self.board) <= 1:
            cap = COMPETITION_OPENING_CAP_SEC
        elif danger:
            cap = COMPETITION_DANGER_CAP_SEC
        else:
            cap = COMPETITION_NORMAL_CAP_SEC

        if danger and remaining_seconds > COMPETITION_FAST_THRESHOLD_SEC:
            fair_share *= COMPETITION_DANGER_MULTIPLIER

        hard_cap = max(0.05, remaining_seconds - COMPETITION_HARD_STOP_MARGIN_SEC)
        budget = min(cap, fair_share, hard_cap)
        return max(COMPETITION_MIN_MOVE_BUDGET_SEC, budget)

    def choose_move(self, full_input):
        if is_full_input(full_input):
            self._load_from_full_input(full_input)
        else:
            self._apply_incremental_request(full_input)

        move, debug = self.choose_internal_move()
        if move is None:
            return EMERGENCY_MOVE_BOT, debug
        bot_move = internal_to_bot(move[0], move[1])
        return bot_move, debug


def make_response(move, debug):
    return json.dumps({"response": {"x": move[0], "y": move[1]}, "debug": debug}, separators=(",", ":"))


def emergency_move(full_input):
    try:
        if is_full_input(full_input):
            position = reconstruct_position(full_input)
            board = position["board"]
            legal = get_legal_moves(board)
            if legal:
                row, col = best_center_tiebreak(legal)
                return internal_to_bot(row, col)
        else:
            move = parse_move(full_input)
            if move is not None:
                x, y = move
                candidates = []
                for row in range(BOARD_SIZE):
                    for col in range(BOARD_SIZE):
                        bx, by = internal_to_bot(row, col)
                        if (bx, by) != (x, y):
                            candidates.append((row, col))
                if candidates:
                    row, col = best_center_tiebreak(candidates)
                    return internal_to_bot(row, col)
    except Exception:
        pass
    return EMERGENCY_MOVE_BOT


STANDARD_COLUMNS = "ABCDEFGHIJK"
STANDARD_COLOR_TO_STONE = {"R": BLACK, "B": WHITE}
STANDARD_STONE_TO_COLOR = {BLACK: "R", WHITE: "B"}
STANDARD_MOVE_RE = re.compile(
    r"^\s*(?:(?P<color>[RB])\s*)?(?:\(\s*)?"
    r"(?P<column>[A-K])(?:\s*,\s*|\s*)"
    r"(?P<row>1[01]|[1-9])\s*\)?\s*$",
    re.IGNORECASE,
)


def parse_standard_move(text, expected_stone=None):
    """Parse official Hex notation into internal (row, col) coordinates.

    Accepted forms are ``E7``, ``E,7``, ``(E,7)`` and the record notation
    ``R(E,7)``/``B(E,7)``.  The engine's BLACK/WHITE axes were trained with
    the first player joining the left/right edges.  The official record uses
    R for the top/bottom player and B for the left/right player, so the CLI
    applies a transpose: standard column -> internal row, standard row ->
    internal column.
    """
    match = STANDARD_MOVE_RE.match(text or "")
    if not match:
        raise ValueError("move must look like E7 or R(E,7), with A-K and 1-11")

    color = match.group("color")
    color = color.upper() if color else None
    if color is not None:
        stone = STANDARD_COLOR_TO_STONE[color]
        if expected_stone is not None and stone != expected_stone:
            expected_color = STANDARD_STONE_TO_COLOR[expected_stone]
            raise ValueError("expected a {0} move".format(expected_color))

    column = match.group("column").upper()
    row_number = int(match.group("row"))
    return STANDARD_COLUMNS.index(column), row_number - 1


def format_standard_move(move, stone):
    if move is None:
        return ""
    row, col = move
    if not inside(row, col):
        raise ValueError("internal move is outside the Hex board")
    return "{0}({1},{2})".format(
        STANDARD_STONE_TO_COLOR[stone], STANDARD_COLUMNS[row], col + 1
    )


def print_competition_debug(move_text, debug):
    """Keep diagnostics off stdout so stdout remains easy to copy as a move."""
    if not debug:
        return
    parts = ["[debug]", move_text]
    for key in (
        "mode", "ply", "think_seconds", "budget_seconds", "bot_total_seconds",
        "remaining_seconds", "sims", "best_n", "root_v", "best_q", "path",
        "model_error",
    ):
        if key in debug:
            parts.append("{0}={1}".format(key, debug[key]))
    print(" ".join(parts), file=sys.stderr, flush=True)


class CompetitionTimeLogger(object):
    """CSV timing log for our bot; separate from the official棋谱 txt."""

    FIELDNAMES = (
        "round", "ply", "side", "move", "think_seconds", "bot_total_seconds",
        "remaining_seconds", "budget_seconds", "mode"
    )

    def __init__(self, path):
        self.path = os.path.abspath(path) if path else ""
        self.handle = None
        self.writer = None
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.handle = open(self.path, "w", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()
        self.handle.flush()

    def record(
        self, ply, side, move_text, think_seconds, bot_total_seconds,
        remaining_seconds, budget_seconds, mode
    ):
        if self.writer is None:
            return
        self.writer.writerow({
            "round": (ply + 1) // 2,
            "ply": ply,
            "side": side,
            "move": move_text,
            "think_seconds": "{0:.6f}".format(think_seconds),
            "bot_total_seconds": "{0:.6f}".format(bot_total_seconds),
            "remaining_seconds": "{0:.6f}".format(remaining_seconds),
            "budget_seconds": "{0:.6f}".format(budget_seconds),
            "mode": mode,
        })
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class CompetitionSession(object):
    """Long-lived manual match adapter for the official Hex notation."""

    def __init__(self, bot, time_log_path=None):
        self.bot = bot
        self.time_logger = CompetitionTimeLogger(time_log_path)
        self.started = False
        self.game_over = False
        self.next_ply = 1
        self.bot_total_seconds = 0.0

    def close(self):
        self.time_logger.close()

    def _record_bot_move(
        self, side, move_text, think_seconds, remaining_seconds, budget_seconds, mode
    ):
        ply = self.next_ply
        self.next_ply += 1
        self.bot_total_seconds += think_seconds
        self.time_logger.record(
            ply, side, move_text, think_seconds, self.bot_total_seconds,
            remaining_seconds, budget_seconds, mode
        )
        return ply

    @property
    def side_color(self):
        if self.bot.our_stone is None:
            return "?"
        return STANDARD_STONE_TO_COLOR[self.bot.our_stone]

    @property
    def opponent_color(self):
        if self.bot.opp_stone is None:
            return "?"
        return STANDARD_STONE_TO_COLOR[self.bot.opp_stone]

    def start(self, color):
        color = (color or "").strip().upper()
        if color not in STANDARD_COLOR_TO_STONE:
            raise ValueError("side must be R (first) or B (second)")

        self.bot.board = empty_board()
        self.bot.our_stone = STANDARD_COLOR_TO_STONE[color]
        self.bot.opp_stone = opponent_of(self.bot.our_stone)
        self.bot.forced = None
        self.bot.last_opp_move = None
        self.bot.initialized = True
        self.started = True
        self.game_over = False
        self.next_ply = 1
        self.bot_total_seconds = 0.0

        outputs = ["SIDE {0}".format(color)]
        if self.bot.our_stone == BLACK:
            outputs.append(self._choose_and_format())
        else:
            outputs.append("WAIT {0}".format(self.opponent_color))
        return outputs

    def _choose_and_format(self):
        danger = min_connection_cost(self.bot.board, self.bot.opp_stone) <= 4
        budget_seconds = self.bot.competition_time_budget(
            self.bot_total_seconds, danger=danger
        )
        decision_started = time.perf_counter()
        move, debug = self.bot.choose_internal_move(time_budget_sec=budget_seconds)
        think_seconds = time.perf_counter() - decision_started
        remaining_after = max(
            0.0,
            COMPETITION_TOTAL_TIME_SEC - self.bot_total_seconds - think_seconds,
        )
        if move is None:
            self.game_over = True
            return "GAME_OVER board-is-full"

        move_text = format_standard_move(move, self.bot.our_stone)
        ply = self._record_bot_move(
            self.side_color,
            move_text,
            think_seconds,
            remaining_after,
            budget_seconds,
            mode=debug.get("mode", "bot"),
        )
        debug["ply"] = ply
        debug["think_seconds"] = round(think_seconds, 4)
        debug["budget_seconds"] = round(budget_seconds, 4)
        debug["bot_total_seconds"] = round(self.bot_total_seconds, 4)
        debug["remaining_seconds"] = round(remaining_after, 4)
        print_competition_debug(move_text, debug)
        if has_winning_path(self.bot.board, self.bot.our_stone):
            self.game_over = True
        return move_text

    def receive_opponent_move(self, text):
        if not self.started:
            raise ValueError("enter start R or start B before entering moves")
        if self.game_over:
            raise ValueError("game is already over; enter start R or start B")

        move = parse_standard_move(text, expected_stone=self.bot.opp_stone)
        if self.bot.board[move[0]][move[1]] != EMPTY:
            raise ValueError("that Hex cell is already occupied")

        apply_internal_move(self.bot.board, move, self.bot.opp_stone)
        self.bot.last_opp_move = move
        opponent_move_text = format_standard_move(move, self.bot.opp_stone)
        self.next_ply += 1
        print("[input] {0}".format(opponent_move_text), file=sys.stderr, flush=True)

        if has_winning_path(self.bot.board, self.bot.opp_stone):
            self.game_over = True
            return ["GAME_OVER winner={0}".format(self.opponent_color)]

        return [self._choose_and_format()]


def run_competition_cli(bot, initial_side=None, time_log_path=None):
    if not bot.model.ready:
        detail = bot.model.error or "unknown model loading error"
        raise RuntimeError(
            "competition mode requires a loaded neural-network weight; {0}".format(detail)
        )
    session = CompetitionSession(bot, time_log_path)
    model_name = bot.model.kind
    model_file = os.path.basename(bot.model.path) if bot.model.path else "none"
    try:
        if initial_side:
            print("READY HEX-CLI model={0} weight={1} side={2}".format(
                model_name, model_file, initial_side.upper()
            ), flush=True)
        else:
            print("READY HEX-CLI model={0} weight={1} enter=start R/B".format(
                model_name, model_file
            ), flush=True)
        if session.time_logger.path:
            print("TIME_LOG {0}".format(session.time_logger.path), flush=True)

        if initial_side:
            for output in session.start(initial_side):
                print(output, flush=True)

        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            command = line.upper()
            if command in ("QUIT", "EXIT"):
                print("BYE", flush=True)
                break
            if command in ("HELP", "?"):
                print(
                    "start R|B; moves: E7, E,7, (E,7), R(E,7); "
                    "--time-log records bot think seconds only; quit",
                    flush=True,
                )
                continue

            try:
                fields = line.split()
                if fields and fields[0].upper() in ("START", "NEW", "RESET"):
                    if len(fields) != 2:
                        raise ValueError("use start R or start B")
                    outputs = session.start(fields[1])
                else:
                    outputs = session.receive_opponent_move(line)
                for output in outputs:
                    print(output, flush=True)
            except Exception as exc:
                print("ERROR {0}".format(str(exc)), flush=True)
    finally:
        session.close()


def run_botzone(bot):
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        debug = {}
        try:
            full_input = json.loads(line)
            move, debug = bot.choose_move(full_input)
        except Exception as exc:
            try:
                full_input = json.loads(line)
            except Exception:
                full_input = {}
            move = emergency_move(full_input)
            debug = {
                "mode": "exception",
                "error": str(exc)[:160],
                "trace": traceback.format_exc().splitlines()[-1][:160],
            }

        sys.stdout.write(make_response(move, debug) + "\n")
        if ENABLE_KEEP_RUNNING:
            sys.stdout.write(KEEP_RUNNING_SENTINEL + "\n")
        sys.stdout.flush()
        if not ENABLE_KEEP_RUNNING:
            break


def require_competition_model(bot):
    """Fail closed for a real match instead of silently playing heuristically."""
    if not bot.model.ready or not bot.model.path:
        detail = bot.model.error or "model file was not loaded"
        raise RuntimeError("competition mode requires loaded neural weights: {0}".format(detail))


def main():
    parser = argparse.ArgumentParser(
        description="Hex neural bot: official manual CLI by default, Botzone JSON with --botzone"
    )
    parser.add_argument(
        "--botzone",
        action="store_true",
        help="use the legacy Botzone JSON protocol instead of the competition CLI",
    )
    parser.add_argument(
        "--side",
        choices=("R", "B", "r", "b"),
        help="start a competition game immediately as red/first or blue/second",
    )
    parser.add_argument(
        "--time-log",
        metavar="PATH",
        help="write our bot's per-ply think timing to a UTF-8 CSV file",
    )
    args = parser.parse_args()

    bot = HexBot()
    if args.botzone:
        run_botzone(bot)
    else:
        try:
            require_competition_model(bot)
        except RuntimeError as exc:
            print("FATAL {0}".format(exc), file=sys.stderr, flush=True)
            return 2
        run_competition_cli(bot, args.side, args.time_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
