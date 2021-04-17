import sys
import numpy as np
import pygame
import h5py
import gameboardClass
import agentClass
from params import param_dict

# Command line args, currently reads first argument as parameter set string
if len(sys.argv):
    if sys.argv[1] not in param_dict:
        print("Invalid parameter set, exiting.")
        exit()
    else:
        param_set = param_dict[sys.argv[1]]
        strategy_file = param_set['strategy_file']
        human_player = param_set['human_player']
        evaluate_agent = param_set['evaluate_agent']

    if strategy_file:
        human_player = True
        evaluate_agent = True
else:
    # Choose to control the game yourself ('human_player=1') to test the setups in the different tasks
    param_set=params['1a']
    human_player=True
    evaluate_agent = False

if not human_player or evaluate_agent:
    if not param_set['use_deepq']:
        agent=agentClass.TQAgent(param_set)
    else:
        agent=agentClass.TDQNAgent(param_set)

# The remaining code below is implementation of the game. You don't need to change anything below this line
if evaluate_agent:
    agent_evaluate=agent;
if human_player:
    agent=agentClass.THumanAgent()

gameboard=gameboardClass.TGameBoard(
    param_set['N_row'],
    param_set['N_col'],
    param_set['tile_size'],
    param_set['max_tile_count'],
    agent,
    param_set['stochastic_prob'],
    param_set['name'])

if evaluate_agent:
    agent_evaluate.epsilon=0
    agent_evaluate.fn_init(gameboard, param_set['name'])
    agent_evaluate.fn_load_strategy(param_set['strategy_file'])

if isinstance(gameboard.agent,agentClass.THumanAgent):
    # The player is human

    # Define some colors for painting
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREY = (128, 128, 128)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED =  (255, 0, 0)

    # Initialize the game engine
    pygame.init()
    N_col = param_set['N_col']
    N_row = param_set['N_row']
    screen=pygame.display.set_mode((200+N_col*20,150+N_row*20))
    clock=pygame.time.Clock()
    pygame.key.set_repeat(300,100)
    pygame.display.set_caption('Turn-based tetris')
    font=pygame.font.SysFont('Calibri',25,True)
    fontLarge=pygame.font.SysFont('Calibri',50,True)
    framerate=0;

    # Loop until the window is closed
    while True:
        if isinstance(gameboard.agent,agentClass.THumanAgent):
            gameboard.agent.fn_turn(pygame)
        else:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_SPACE:
                        if framerate > 0:
                            framerate=0
                        else:
                            framerate=10
                    if (event.key==pygame.K_LEFT) and (framerate>1):
                        framerate-=1
                    if event.key==pygame.K_RIGHT:
                        framerate+=1
                        gameboard.agent.fn_turn()

        if evaluate_agent:
            agent_evaluate.fn_read_state()
            agent_evaluate.fn_select_action()

        if pygame.display.get_active():
            # Paint game board
            screen.fill(COLOR_WHITE)

            for i in range(gameboard.N_row):
                for j in range(gameboard.N_col):
                    pygame.draw.rect(screen,COLOR_GREY,[100+20*j,80+20*(gameboard.N_row-i),20,20],1)
                    if gameboard.board[i][j] > 0:
                        pygame.draw.rect(screen,COLOR_BLACK,[101+20*j,81+20*(gameboard.N_row-i),18,18])

            if gameboard.cur_tile_type is not None:
                curTile=gameboard.tiles[gameboard.cur_tile_type][gameboard.tile_orientation]
                for xLoop in range(len(curTile)):
                    for yLoop in range(curTile[xLoop][0],curTile[xLoop][1]):
                        pygame.draw.rect(screen,COLOR_RED,[101+20*((xLoop+gameboard.tile_x)%gameboard.N_col),81+20*(gameboard.N_row-(yLoop+gameboard.tile_y)),18,18])

            screen.blit(font.render("Reward: "+str(agent.reward_tots[agent.episode]),True,COLOR_BLACK),[0,0])
            screen.blit(font.render("Tile "+str(gameboard.tile_count)+"/"+str(gameboard.max_tile_count),True,COLOR_BLACK),[0,20])
            if framerate>0:
                screen.blit(font.render("FPS: "+str(framerate),True,COLOR_BLACK),[320,0])
                screen.blit(font.render("Reward: "+str(agent.reward_tots[agent.episode]),True,COLOR_BLACK),[0,0])
            if gameboard.gameover:
                screen.blit(fontLarge.render("Game Over", True,COLOR_RED), [80, 200])
                screen.blit(font.render("Press ESC to try again", True,COLOR_RED), [85, 265])

            pygame.display.flip()
            clock.tick(framerate)
else:
    # The player is AI
    while True:
        gameboard.agent.fn_turn()


