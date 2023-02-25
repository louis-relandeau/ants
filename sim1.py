import numpy as np
import pygame, time, cv2, random, threading, copy, math

class Viewer:
    def __init__(self, update_func, display_size):
        self.update_func = update_func
        self.display_size = display_size
        pygame.init()
        self.display = pygame.display.set_mode(self.display_size)
    
    def start(self):
        paths_to_food = np.float32(np.zeros((size, size)))
        paths_to_home = np.float32(np.zeros((size, size)))
        food_image = np.float32(np.zeros((size, size)))
        home_image = np.float32(np.zeros((size, size)))
        for food in food_sources:
            cv2.circle(food_image, (int(food.pos[0])%size, int(food.pos[1])%size), food.radius, 1, thickness=-1)
        cv2.circle(home_image, (int(home_pos[0])%size, int(home_pos[1])%size), home_rad, 1, thickness=-1)

        running = True
        step_count = 0
        start_time = time.perf_counter()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            display_image = np.ones((size, size, 3)) * 255.0
            display_image = self.addToImage(display_image, home_image, (0,0,0))
            display_image = self.addToImage(display_image, food_image, (0,0,255))
            paths_to_food, paths_to_home, ants = self.update_func(paths_to_food, paths_to_home)
            display_image = self.addToImage(display_image, paths_to_food, (0,255,0))
            display_image = self.addToImage(display_image, paths_to_home, (255,0,0))
            display_image = self.addToImage(display_image, ants, (0,0,0))
            # pad = 1
            # pad_width = ((pad, pad), (pad, pad), (0, 0))
            # display_image = np.pad(display_image, pad_width=pad_width, mode='constant', constant_values=255)
            surf = pygame.surfarray.make_surface(display_image.astype('uint8'))
            self.display.blit(surf, (0, 0))

            pygame.display.update()
            step_count += 1
        print("--- {} ms per step ---".format((time.perf_counter() - start_time)/step_count*1000))
        pygame.quit()

    def addToImage(self, image, mask, color):
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # convert mask to a 3-channel image
        mask = mask.astype(bool)  # convert to boolean mask

        # create a color array of the same shape as the original image
        color_array = np.zeros_like(image)
        color_array[:] = color

        # set the color of the masked pixels in the original image
        image[mask] = color_array[mask]

        return image

def updateParticles(to_food, to_home):

    ants_image = np.float32(np.zeros((size, size)))

    to_food = np.where(to_food > 0, to_food-0.007, 0)
    to_home = np.where(to_home > 0, to_home-0.007, 0)
    for ant in colony:
        pos = (int(ant.pos[0])%size, int(ant.pos[1])%size)
        cv2.circle(ants_image, pos, 5, 1, thickness=-1)

        dist = 10
        width = 0.2*math.pi
        view_range = {0: [(ant.pos[0]+np.cos(ant.ang)*dist)%size, (ant.pos[1]+np.sin(ant.ang)*dist)%size]}
        for alpha in np.arange(-width,width,0.05):
            view_range[alpha] = [(ant.pos[0]+np.cos(ant.ang+alpha)*dist)%size, (ant.pos[1]+np.sin(ant.ang+alpha)*dist)%size]
        # for view in view_range.values():
        #     ants_image[int(view[1])%size, int(view[0])%size] = 0.5

        in_home = False
        if np.linalg.norm(np.array(home_pos) - ant.pos) < home_rad:
            in_home = True

        if in_home:
            # if ant.carries_food:
            #     ant.ang = (ant.ang + math.pi) % (2*math.pi)
            ant.carries_food = False
            ant.previously_found = None
            ant.updatePos(view_range, to_home)
            ant.cooldown = time.time()
        else:
            in_food = False
            for food in food_sources:
                if np.linalg.norm(food.pos - ant.pos) < food.radius:
                    in_food = True
                    break
            if in_food:
                # if not ant.carries_food:
                # ant.ang = (ant.ang + math.pi) % (2*math.pi)
                ant.carries_food = True
                ant.previously_found = time.time()
                ant.updatePos(view_range, to_food)
                ant.cooldown = time.time()
            elif ant.carries_food:
                ant.updatePos(view_range, to_home)
            else:
                ant.updatePos(view_range, to_food)
        
        if ant.cooldown:
            if ant.carries_food:
                to_food[pos[1], pos[0]] = 1
            else:
                to_home[pos[1], pos[0]] = 1
            if (time.time()-ant.cooldown) > COOLDOWN_TIME:
                ant.cooldown = None
            
    return to_food, to_home, ants_image

class Ant:
    def __init__(self, position) -> None:
        self.pos = np.array(position)
        self.vel = 1
        self.ang = np.random.uniform(0, 2*math.pi)
        # self.ang = math.pi / 4
        self.dt = 2
        self.carries_food = False
        self.cooldown = None

    def updatePos(self, points, path_image):
        # self.pos = (self.pos + np.array([self.vel*self.dt*np.cos(self.ang), self.vel*self.dt*np.sin(self.ang)]))%size
        self.pos += np.array([self.vel*self.dt*np.cos(self.ang), self.vel*self.dt*np.sin(self.ang)])
        edge_offset = 10
        if any([val < edge_offset or val >= size-edge_offset for val in self.pos]):
            self.pos = np.clip(self.pos, edge_offset, size-edge_offset)
            self.ang = (self.ang + math.pi) % 2 * math.pi
        
        all_angles = []
        for [angle, point] in points.items():
            val = int(path_image[int(point[1]), int(point[0])]*10)
            for _ in range(val):
                all_angles.append(angle)
        if all_angles:
            self.ang += np.mean(all_angles)

        a = 0.1
        v = 0.1
        b = random.sample([-a,a], k=1)[0]
        c = random.gauss(b,v)
        self.ang = (self.ang + c) % (2 * math.pi)
    
class FoodSource:
    def __init__(self, pos, radius) -> None:
        self.pos = np.array(pos)
        self.radius = radius

if __name__ == "__main__":
    size = 300
    COOLDOWN_TIME = 10

    home_pos = [size/2, size/2]
    home_rad = 30
    
    colony = []
    for _ in range(250):
        colony.append(Ant(home_pos))
        # colony.append(Ant(np.random.uniform(0, size, 2)))

    food_sources = []
    offset = 130
    food_sources.append(FoodSource([size/2-offset,size/2-offset],home_rad//2))
    food_sources.append(FoodSource([size/2+offset,size/2-offset],home_rad//2))
    food_sources.append(FoodSource([size/2-offset,size/2+offset],home_rad//2))
    food_sources.append(FoodSource([size/2+offset,size/2+offset],home_rad//2))

    viewer = Viewer(updateParticles, (size+2, size+2))
    viewer.start()
