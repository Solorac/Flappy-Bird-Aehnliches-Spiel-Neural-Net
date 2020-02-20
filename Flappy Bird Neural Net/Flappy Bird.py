import pygame
import neat
import os
import random

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 500
GRAVITY = 1

screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
clock = pygame.time.Clock()
FPS = 240 # Frames per second.

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 32
        self.velocity = 0
        self.rect = pygame.Rect((self.x, self.y), (self.width, self.width))

    def jump(self):
        self.velocity = -3

    def move(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        if self.y <= 0:
            self.y = 0
            self.velocity = 0

        self.rect.move_ip(0, self.velocity)

    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)


class Pipe:
    GAP = 64
    VELOCITY = 1

    def __init__(self):
        self.x = WIN_WIDTH
        self.passed = False

        self.width = 32
        self.height = 0
        self.bottom = 0
        self.top = 0
        self.set_height()

        self.rect_top = pygame.Rect((self.x, self.top), (self.width, self.height))
        self.rect_bottom = pygame.Rect((self.x, self.bottom), (self.width, WIN_HEIGHT - self.bottom))

    def set_height(self):
        self.height = random.randrange(10, 500-self.GAP-10)
        self.bottom = self.height + self.GAP
        self.top = 0

    def move(self):
        self.x -= self.VELOCITY
        self.rect_top.move_ip(-self.VELOCITY, 0)
        self.rect_bottom.move_ip(-self.VELOCITY, 0)

    def draw(self):
        pygame.draw.rect(screen, GREEN, self.rect_top)
        pygame.draw.rect(screen, GREEN, self.rect_bottom)

    def collide(self, bird):
        return self.rect_top.colliderect(bird.rect) or self.rect_bottom.colliderect(bird.rect)


def draw_window(birds, pipes, score):
    screen.fill(BLACK)
    for pipe in pipes:
        pipe.draw()
    for bird in birds:
        bird.draw()
    text = STAT_FONT.render("Score: " + str(score), 1, WHITE)
    screen.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))


def main(genomes, config):
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(100, 100))
        g.fitness = 0
        ge.append(g)

    pipes = [Pipe()]
    score = 0
    run = True

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].width:
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()

        rem = []
        add_pipe = False

        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird) or bird.y > WIN_HEIGHT-bird.width:
                    ge[x].fitness -= 10
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.width < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            pipes.append(Pipe())

        for r in rem:
            pipes.remove(r)

        draw_window(birds, pipes, score)
        pygame.display.update()

        if score > 300:
            break

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter((neat.StdOutReporter(True)))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 5000)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
