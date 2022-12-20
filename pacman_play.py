import argparse
import pygame as pg
import pygame_menu
import sys

from pygame_menu.themes import Theme
from game import Game
from map import Map


class Controller(object):
    """
    control the operation of game, such as start, finsh, restart and so on.
    """

    def __init__(self, layout_name: str, act_sound: bool, act_state: bool, **kwargs):
        pg.init()
        self.layout_name = layout_name
        self.act_sound = act_sound
        self.act_state = act_state
        self.ai_agent = kwargs['ai_agent']
        self.maze = Map(layout_name)
        self.width, self.height = self.maze.get_map_sizes()
        self.screen = Controller.get_screen(act_state, self.width, self.height)
        self.menu_theme = Theme(
            title_font=pygame_menu.font.FONT_NEVIS,
            widget_font=pygame_menu.font.FONT_NEVIS,
            background_color=(120, 220, 220),
        )
        pg.display.set_caption("Pacman")

    def load_level(self):
        game = Game(
            maze=self.maze,
            screen=self.screen,
            sounds_active=self.act_sound,
            state_active=self.act_state,
            agent=self.ai_agent
        )
        game.start_game()

    def change_sound(self, state, value):
        if state == "open":
            self.act_sound = True
        else:
            self.act_sound = False

    @staticmethod
    def get_screen(act_state: bool, width: int, height: int) -> pg.SurfaceType:
        if act_state:
            return pg.display.set_mode(((width * 2) + 48, height))
        else:
            return pg.display.set_mode((width, height))

    def load_menu(self):
        # modify variable name
        game_menu = pygame_menu.Menu(self.height, self.width,
                                'Welcome to Pacman', theme=self.menu_theme)

        game_menu.add_button('Play', self.load_level)
        game_menu.add_button('Quit', pygame_menu.events.EXIT)

        # add sound change button
        game_menu.add_selector('Sound', [('open', 1), ('close', 2)],
                          onchange=self.change_sound)
        game_menu.mainloop(self.screen)


def main(args):
    controller = Controller(layout_name=args.layout[0], act_sound=args.sound, act_state=args.state, ai_agent=None)
    controller.load_menu()


def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the Pacman Game')
    parser.add_argument('-lay', '--layout', type=str, nargs=1, default=["classic"],
                        help="Name of layout to load in the game")
    parser.add_argument('-snd', '--sound', action='store_true', default=True,
                        help="Activate sounds in the game")
    parser.add_argument('-stt', '--state', action='store_true',
                        help="Display the state matrix of the game")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
    pg.quit()
    sys.exit()
