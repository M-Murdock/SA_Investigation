import pygame
import sys

class Method_Selector:
    def __init__(self, options=["Policy A", "Policy B", "Policy C", "Policy D"], caption="Option Selector"):
        pygame.init()
        pygame.font.init()

        # ------------ CONFIG ------------
        self.OPTIONS = options
        WIDTH, HEIGHT = 400, 400
        self.FONT = pygame.font.SysFont(None, 36)
        self.BG_COLOR = (30, 30, 30)
        self.TEXT_COLOR = (220, 220, 220)
        self.HIGHLIGHT_COLOR = (70, 130, 180)
        self.HOVER_COLOR = (120, 120, 120)   
        # --------------------------------

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(caption)

        self.selected_index = None
        self.selected_option = None
        self.hover_index = None

    def draw_options(self):
        self.screen.fill(self.BG_COLOR)
        y = 60
        for i, option in enumerate(self.OPTIONS):
            self.rect = pygame.Rect(50, y - 5, 300, 40)
            
            if i == self.selected_index:
                pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, (50, y - 5, 300, 40), border_radius=6)
                # Hover highlight (only if not selected)
            if i == self.hover_index and i != self.selected_index:
                pygame.draw.rect(self.screen, self.HOVER_COLOR, self.rect, border_radius=6)

            # Selected highlight
            if i == self.selected_index:
                pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, self.rect, border_radius=6)
            
            # Draw text
            text = self.FONT.render(option.value, True, self.TEXT_COLOR)
            self.screen.blit(text, (60, y))
            y += 50
        pygame.display.flip()

    def get(self):
        running = True

        while running:
            # --- Mouse position detection for hover ---
            mx, my = pygame.mouse.get_pos()
            self.hover_index = None
            for i in range(len(self.OPTIONS)):
                text_y = 60 + 50 * i
                if 50 < mx < 350 and text_y - 5 < my < text_y + 35:
                    self.hover_index = i
            # ---- DRAW ----
            self.draw_options()

            # ---- EVENTS ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    for i in range(len(self.OPTIONS)):
                        text_y = 60 + 50 * i
                        if 50 < x < 350 and text_y - 5 < y < text_y + 35:
                            selected_index = i
                            selected_option = self.OPTIONS[i]

                            print("Final selection ->", selected_option)
                            return selected_option
                            

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and selected_index is not None:
                        running = False

        # Once we exit the loop, Pygame is done
        pygame.quit()
        # print("Final selection ->", selected_option)
        sys.exit()
        return -1
