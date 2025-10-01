import pygame
import random
import asyncio
import os
import math
import copy

pygame.init()
pygame.font.init()

# --- 日本語フォント指定 ---
def load_jp_font(size):
    try:
        font_path = os.path.join("assets", "NotoSansJP-Regular.ttf")
        if os.path.exists(font_path):
            return pygame.font.Font(font_path, size)
        else:
            return pygame.font.Font("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", size)
    except:
        return pygame.font.SysFont(None, size)

font = load_jp_font(40)
big_font = load_jp_font(80)

# --- 定数 ---
BOARD_SIZE = 8
CELL_SIZE = 80
WIDTH, HEIGHT = BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE + 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
HIGHLIGHT = (200, 200, 0)

# --- ゲーム状態管理 ---
game_state = "select_level"
cpu_level = "normal"

IGNORE_MS = 350
ignore_until = 0

# --- 盤面データ ---
board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
board[3][3], board[4][4] = 2, 2
board[3][4], board[4][3] = 1, 1

current_player = 1
passes = 0
game_over = False

# --- ウィンドウ ---
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("オセロ")

DIRECTIONS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == dy == 0)]

# --- 盤面描画 ---
def draw_board(valid_moves=[]):
    screen.fill(GREEN)
    for x in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (x * CELL_SIZE, 0), (x * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
    for y in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (0, y * CELL_SIZE), (BOARD_SIZE * CELL_SIZE, y * CELL_SIZE))
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != 0:
                color = BLACK if board[y][x] == 1 else WHITE
                pygame.draw.circle(screen, color,
                                   (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2),
                                   CELL_SIZE//2 - 5)
    for (x, y) in valid_moves:
        pygame.draw.rect(screen, HIGHLIGHT, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE), 5)

def is_valid_move(x, y, player, state=None):
    b = state if state else board
    if b[y][x] != 0:
        return False
    opponent = 2 if player == 1 else 1
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        flipped = False
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and b[ny][nx] == opponent:
            nx += dx
            ny += dy
            flipped = True
        if flipped and 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and b[ny][nx] == player:
            return True
    return False

def place_stone(x, y, player, state=None):
    b = state if state else board
    b[y][x] = player
    opponent = 2 if player == 1 else 1
    for dx, dy in DIRECTIONS:
        stones_to_flip = []
        nx, ny = x + dx, y + dy
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and b[ny][nx] == opponent:
            stones_to_flip.append((nx, ny))
            nx += dx
            ny += dy
        if stones_to_flip and 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and b[ny][nx] == player:
            for fx, fy in stones_to_flip:
                b[fy][fx] = player

def get_valid_moves(player, state=None):
    b = state if state else board
    moves = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if is_valid_move(x, y, player, b):
                moves.append((x, y))
    return moves

def count_stones(state=None):
    b = state if state else board
    black = sum(row.count(1) for row in b)
    white = sum(row.count(2) for row in b)
    return black, white

# --- CPUレベル表示 ---
cpu_level_labels = {
    "weak": "弱い",
    "normal": "普通",
    "strong": "強い",
    "oni": "鬼"
}

small_font = load_jp_font(33)

def draw_score(turn_text=""):
    black, white = count_stones()
    text = f"黒: {black}   白: {white}   {turn_text}"
    score_surface = font.render(text, True, BLACK, WHITE)
    screen.blit(score_surface, (10, BOARD_SIZE * CELL_SIZE + 2))

    if cpu_level in cpu_level_labels:
        cpu_text = f"CPU: {cpu_level_labels[cpu_level]}"
        cpu_surface = small_font.render(cpu_text, True, BLACK, WHITE)
        screen.blit(cpu_surface, (WIDTH - cpu_surface.get_width() - 10, BOARD_SIZE * CELL_SIZE + 5))    

def reset_game():
    global board, current_player, passes, game_over
    board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    board[3][3], board[4][4] = 2, 2
    board[3][4], board[4][3] = 1, 1
    current_player = 1
    passes = 0
    game_over = False

# --- 勝敗表示 ---
def show_winner(button_pressed=False):
    black, white = count_stones()
    if black > white:
        msg = "黒の勝ち！"
    elif white > black:
        msg = "白の勝ち！"
    else:
        msg = "引き分け！"

    result_surface = big_font.render(msg, True, BLACK, WHITE)
    rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    screen.blit(result_surface, rect)

    button_color = (0, 100, 0) if not button_pressed else (0, 70, 0)
    offset_y = 3 if button_pressed else 0

    button_text = font.render("もう一度プレイ！", True, WHITE)
    button_rect = button_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 90 + offset_y))
    rect_draw = button_rect.inflate(40, 20)
    rect_draw.center = (WIDTH // 2, HEIGHT // 2 + 90 + offset_y)
    pygame.draw.rect(screen, button_color, rect_draw, border_radius=12)

    screen.blit(button_text, button_text.get_rect(center=rect_draw.center))
    return rect_draw

# --- 評価関数 ---
def evaluate(state, player):
    opponent = 2 if player == 1 else 1
    black, white = count_stones(state)
    my_score = black if player == 1 else white
    opp_score = white if player == 1 else black

    score = my_score - opp_score
    my_moves = len(get_valid_moves(player, state))
    opp_moves = len(get_valid_moves(opponent, state))
    mobility = my_moves - opp_moves

    corners = [(0,0),(0,7),(7,0),(7,7)]
    my_corners = sum(1 for (x,y) in corners if state[y][x]==player)
    opp_corners = sum(1 for (x,y) in corners if state[y][x]==opponent)
    corner_score = 25*(my_corners - opp_corners)

    return score + mobility*2 + corner_score*5

# --- ミニマックス探索 ---
def minimax(state, depth, alpha, beta, maximizing, player):
    opponent = 2 if player == 1 else 1
    moves = get_valid_moves(player if maximizing else opponent, state)

    if depth == 0 or not moves:
        return evaluate(state, player), None

    if maximizing:
        max_eval = -math.inf
        best_move = None
        for move in moves:
            new_state = copy.deepcopy(state)
            place_stone(move[0], move[1], player, new_state)
            eval, _ = minimax(new_state, depth-1, alpha, beta, False, player)
            if eval > max_eval:
                max_eval, best_move = eval, move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        best_move = None
        for move in moves:
            new_state = copy.deepcopy(state)
            place_stone(move[0], move[1], opponent, new_state)
            eval, _ = minimax(new_state, depth-1, alpha, beta, True, player)
            if eval < min_eval:
                min_eval, best_move = eval, move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# --- CPU の打ち方 ---
def cpu_move(player, moves):
    if cpu_level == "weak":
        best_move = None
        best_gain = -1
        for (x, y) in moves:
            tmp_board = [row[:] for row in board]
            place_stone(x, y, player)
            black, white = count_stones()
            gain = (black if player == 1 else white)
            if gain > best_gain:
                best_gain = gain
                best_move = (x, y)
            for yy in range(BOARD_SIZE):
                board[yy] = tmp_board[yy][:]
        return best_move if best_move else random.choice(moves)

    elif cpu_level == "normal":
        corners = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, 0), (BOARD_SIZE-1, BOARD_SIZE-1)]
        corner_moves = [m for m in moves if m in corners]
        if corner_moves:
            return random.choice(corner_moves)
        best_move = None
        best_gain = -1
        for (x, y) in moves:
            tmp_board = [row[:] for row in board]
            place_stone(x, y, player)
            black, white = count_stones()
            gain = (black if player == 1 else white)
            if gain > best_gain:
                best_gain = gain
                best_move = (x, y)
            for yy in range(BOARD_SIZE):
                board[yy] = tmp_board[yy][:]
        return best_move if best_move else random.choice(moves)

    elif cpu_level == "strong":
        corners = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, 0), (BOARD_SIZE-1, BOARD_SIZE-1)]
        x_sides = [(0, 1), (1, 0), (1, 1),
                   (0, BOARD_SIZE-2), (1, BOARD_SIZE-1), (1, BOARD_SIZE-2),
                   (BOARD_SIZE-2, 0), (BOARD_SIZE-1, 1), (BOARD_SIZE-2, 1),
                   (BOARD_SIZE-2, BOARD_SIZE-1), (BOARD_SIZE-1, BOARD_SIZE-2), (BOARD_SIZE-2, BOARD_SIZE-2)]

        corner_moves = [m for m in moves if m in corners]
        if corner_moves:
            return random.choice(corner_moves)

        safe_moves = [m for m in moves if m not in x_sides]
        target_moves = safe_moves if safe_moves else moves

        best_move = None
        best_gain = -1
        for (x, y) in target_moves:
            tmp_board = [row[:] for row in board]
            place_stone(x, y, player)
            black, white = count_stones()
            gain = (black if player == 1 else white)
            if gain > best_gain:
                best_gain = gain
                best_move = (x, y)
            for yy in range(BOARD_SIZE):
                board[yy] = tmp_board[yy][:]
        return best_move if best_move else random.choice(moves)

    elif cpu_level == "oni":
        _, move = minimax(copy.deepcopy(board), depth=3, alpha=-math.inf, beta=math.inf, maximizing=True, player=player)
        return move if move else random.choice(moves)

    return random.choice(moves)

# --- ボタン描画 ---
def draw_button(text, center, pressed=False, w=200, h=60, base_color=(0, 100, 200)):
    rect = pygame.Rect(0, 0, w, h)
    rect.center = (center[0], center[1] + (3 if pressed else 0))
    color = base_color if not pressed else (base_color[0], max(0, base_color[1]-40), max(0, base_color[2]-50))
    pygame.draw.rect(screen, color, rect, border_radius=12)
    label = font.render(text, True, WHITE)
    screen.blit(label, label.get_rect(center=rect.center))
    return rect

# --- メイン ---
async def main():
    global current_player, passes, game_over, game_state, cpu_level, ignore_until
    clock = pygame.time.Clock()
    running = True
    button_rect = None
    button_pressed = False
    pressed_button_name = None
    prev_game_state = None

    if game_state == "select_level":
        ignore_until = pygame.time.get_ticks() + IGNORE_MS
        pygame.event.clear()

    while running:
        screen.fill(WHITE)

        if game_state != prev_game_state:
            if game_state == "select_level":
                ignore_until = pygame.time.get_ticks() + IGNORE_MS
                pygame.event.clear()
            prev_game_state = game_state

        if game_state == "select_level":
            msg = font.render("CPU の強さを選んでください", True, BLACK)
            screen.blit(msg, msg.get_rect(center=(WIDTH//2, HEIGHT//2-160)))
            weak_btn   = draw_button("弱い", (WIDTH//2, HEIGHT//2-60), pressed_button_name=="weak")
            normal_btn = draw_button("普通", (WIDTH//2, HEIGHT//2+20), pressed_button_name=="normal")
            strong_btn = draw_button("強い", (WIDTH//2, HEIGHT//2+100), pressed_button_name=="strong")
            oni_btn    = draw_button("鬼", (WIDTH//2, HEIGHT//2+180), pressed_button_name=="oni")
            pygame.display.flip()

        elif game_state == "play":
            if not game_over:
                moves = get_valid_moves(current_player)
                if not moves:
                    passes += 1
                    if passes == 2:
                        game_over = True
                    else:
                        current_player = 2 if current_player == 1 else 1
                    await asyncio.sleep(0)
                    continue
                else:
                    passes = 0

                draw_board(moves)
                draw_score("黒の番" if current_player == 1 else "白の番")
                pygame.display.flip()

                if current_player == 2:
                    await asyncio.sleep(0.5)
                    move = cpu_move(2, moves)
                    place_stone(move[0], move[1], 2)
                    current_player = 1
                    continue
            else:
                draw_board()
                draw_score("ゲーム終了")
                button_rect = show_winner(button_pressed)
                pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
            
            if game_state == "select_level" and pygame.time.get_ticks() < ignore_until:
                continue

            if game_state == "select_level":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if weak_btn.collidepoint(mx, my):
                        pressed_button_name = "weak"
                    elif normal_btn.collidepoint(mx, my):
                        pressed_button_name = "normal"
                    elif strong_btn.collidepoint(mx, my):
                        pressed_button_name = "strong"
                    elif oni_btn.collidepoint(mx, my):
                        pressed_button_name = "oni"

                elif event.type == pygame.MOUSEBUTTONUP:
                    mx, my = pygame.mouse.get_pos()
                    if weak_btn.collidepoint(mx, my) and pressed_button_name=="weak":
                        cpu_level = "weak"; game_state = "play"; reset_game()
                    elif normal_btn.collidepoint(mx, my) and pressed_button_name=="normal":
                        cpu_level = "normal"; game_state = "play"; reset_game()
                    elif strong_btn.collidepoint(mx, my) and pressed_button_name=="strong":
                        cpu_level = "strong"; game_state = "play"; reset_game()
                    elif oni_btn.collidepoint(mx, my) and pressed_button_name=="oni":
                        cpu_level = "oni"; game_state = "play"; reset_game()
                    pressed_button_name = None

            elif game_state == "play":
                if (event.type == pygame.MOUSEBUTTONDOWN) and not game_over and current_player == 1:
                    mx, my = pygame.mouse.get_pos()
                    if my < BOARD_SIZE * CELL_SIZE:
                        x, y = mx // CELL_SIZE, my // CELL_SIZE
                        if (x, y) in moves:
                            place_stone(x, y, 1)
                            current_player = 2
                elif game_over:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = pygame.mouse.get_pos()
                        if button_rect and button_rect.collidepoint(mx, my):
                            button_pressed = True
                    elif event.type == pygame.MOUSEBUTTONUP:
                        mx, my = pygame.mouse.get_pos()
                        if button_rect and button_rect.collidepoint(mx, my) and button_pressed:
                            reset_game()
                            game_state = "select_level"
                            ignore_until = pygame.time.get_ticks() + IGNORE_MS
                            pygame.event.clear()
                            button_pressed = False
                            continue
                        button_pressed = False

        clock.tick(60)
        await asyncio.sleep(0)

asyncio.run(main())