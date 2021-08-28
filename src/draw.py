#!/usr/bin/env python3

import pygame

def draw_gameboard(gameboard):
    for i in range(gameboard.N_row):
        for j in range(gameboard.N_col):
            pygame.draw.rect(screen,COLOR_GREY,[100+20*j,80+20*(gameboard.N_row-i),20,20],1)
            if gameboard.board[i][j] > 0:
                pygame.draw.rect(screen,COLOR_BLACK,[101+20*j,81+20*(gameboard.N_row-i),18,18])

def draw_tile(gameboard):
    curTile=gameboard.tiles[gameboard.cur_tile_type][gameboard.tile_orientation]
    for xLoop in range(len(curTile)):
        for yLoop in range(curTile[xLoop][0],curTile[xLoop][1]):
            pygame.draw.rect(screen,COLOR_RED,[101+20*((xLoop+gameboard.tile_x)%gameboard.N_col),81+20*(gameboard.N_row-(yLoop+gameboard.tile_y)),18,18])

def render_text():
    screen.blit(font.render("Reward: "+str(agent.reward_tots[agent.episode]),True,COLOR_BLACK),[0,0])
    screen.blit(font.render("Tile "+str(gameboard.tile_count)+"/"+str(gameboard.max_tile_count),True,COLOR_BLACK),[0,20])
    if framerate>0:
        screen.blit(font.render("FPS: "+str(framerate),True,COLOR_BLACK),[320,0])
        screen.blit(font.render("Reward: "+str(agent.reward_tots[agent.episode]),True,COLOR_BLACK),[0,0])
    if gameboard.gameover:
        screen.blit(fontLarge.render("Game Over", True,COLOR_RED), [80, 200])
        screen.blit(font.render("Press ESC to try again", True,COLOR_RED), [85, 265])
