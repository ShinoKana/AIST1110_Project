# AIST1110_Project
Game: Pacman

Project includes Reinforcement Learning presented on Pacman game built with Pygame.

Install the requirements:

    pip install -r requirements.txt
    pip install -e .

To manually play the game with the classic maze:

    python pacman_play.py -lay classic

Run the game with other layouts:
    
    python pacman_play.py –m human -lay <MAZE_NAME>
    
    # (maze: classic, small, optional; classic by default)
     
e.g.,

    python pacman_play.py -lay optional

    python pacman_play.py –lay small

Run the game and display the state matrix:

    python pacman_play.py -stt

Agent training:

    python pacman_play.py –m <MODE> -lay <MAZE_NAME> -t –e <EPISODES>
    
    #(mode: human, rgb_array, rgb_array by default)
    #(episodes: integer, specify how many episodes you want to train)
    
Agent testing:
     
    python pacman_play.py -lay <MAZE_NAME> -r

Basic interface:

![image](https://user-images.githubusercontent.com/100673497/209434251-702294e0-5d1c-4305-822f-de8149d35946.png)
