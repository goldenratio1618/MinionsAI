#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#define BOARD_SIZE 5

// timeout (in sec)
#define TIMEOUT 10

#define max(x,y) x > y ? x : y

void sig_handler (int signum) {
  _exit(1);
}

typedef struct location {
  int x;
  int y;
} location;

// from https://stackoverflow.com/questions/14579920/fast-sign-of-integer-in-c
int sign(int x) {
  return (x > 0) - (x < 0);
}

int dist(int xi, int yi, int xf, int yf) {
  int dy = abs(yf - yi);
  int dx = abs(xf - xi);
  if ((xi > xf) == (yi > yf)) dx += abs(yf - yi);
  return max(dx, dy);
}

// set adjacent to six adjacent hexes
void get_adjacent_hexes(location * adjacent, location * my_loc) {
  int x = my_loc->x;
  int y = my_loc->y;
  for (int i = 0; i < 6; i ++) {
    adjacent[i].x = x;
    adjacent[i].y = y;
  }
  adjacent[0].x --;
  adjacent[1].x ++;
  adjacent[2].y --;
  adjacent[3].y ++;

  adjacent[4].x ++;
  adjacent[4].y --;

  adjacent[5].x --;
  adjacent[5].y ++;
}

void set_value(int * board, int value) {
  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i ++)
    board[i] = value;
}

int main() {
  char line [256];
  int color;
  location own_captain;
  location enemy_captain;
  location adjacent_hexes[6];
  int graveyards [BOARD_SIZE * BOARD_SIZE];
  int water [BOARD_SIZE * BOARD_SIZE] = {0};
  int money [2];

  // zombies on board (-1 for empty, 0/1 for color, 2 for exhausted zombie of own color)
  int zombies [BOARD_SIZE * BOARD_SIZE];
  // damaged zombies (1 for damaged, 0 for absent or full health)
  int damaged_zombies [BOARD_SIZE * BOARD_SIZE];

  // timer (to determine when Python parent function has died)
  signal(SIGALRM, sig_handler);

  FILE * log;
  log = fopen("log.txt", "w");
  while (1) {
    // reset alarm
    alarm(TIMEOUT);

    // reset graveyards and zombies
    set_value(graveyards, 0);
    set_value(zombies, -1);
    set_value(damaged_zombies, 0);

    line[0] = '\0';
    while (strcmp(line, "Your turn\n") != 0) {
      usleep(10); // sleep for 0.1 ms
      char * captain = " N ";
      char * zombie = " Z ";
      char * graveyard = " True\n";
      fgets(line, 256, stdin);
      //fprintf(log, "%s", line);
      //fflush(log);
      if (strcmp(line, "Your color\n") == 0) {
        fgets(line, 256, stdin);
        color = atoi(line);
        //fprintf(log, "%d\n", color);
      }
      // find location of graveyards
      else if (strstr(line, graveyard) != NULL) {
        int x = (int) line[0] - (int) '0';
        int y = (int) line[2] - (int) '0';
        graveyards[x * BOARD_SIZE + y] = 1;
      }
      // find locations of zombies
      else if (strstr(line, zombie) != NULL) {
        int x = (int) line[0] - (int) '0';
        int y = (int) line[2] - (int) '0';
        int color = (int) line[6] - (int) '0';
        zombies[x * BOARD_SIZE + y] = color;
      }
      // find location of captain and enemy
      else if (strstr(line, captain) != NULL) {
        char x = line[0];
        char y = line[2];
        if (color == (int) line[6] - (int) '0') {
          own_captain.x = (int) x - (int) '0';
          own_captain.y = (int) y - (int) '0';
        } else {
          enemy_captain.x = (int) x - (int) '0';
          enemy_captain.y = (int) y - (int) '0';
        }
      }
      // find money
      else if (strcmp(line, "Money\n") == 0) {
        fgets(line, 256, stdin);
        money[0] = atoi(line);
        fgets(line, 256, stdin);
        money[1] = atoi(line);
      }
      else if (strcmp(line, "Game over!\n") == 0) {
        fclose(log);
        return 0;
      }
    }
    
    // zombie movement
    for (int xi = 0; xi < BOARD_SIZE; xi ++) {
      for (int yi = 0; yi < BOARD_SIZE; yi ++) {
        // see if you have a zombie
        if (zombies [xi * BOARD_SIZE + yi] != color) continue;
        location zombie_loc;
        zombie_loc.x = xi;
        zombie_loc.y = yi;
        get_adjacent_hexes(adjacent_hexes, &zombie_loc);

        // attack damaged enemy zombies
        for (int i = 0; i < 6; i ++) {
          int x = adjacent_hexes[i].x;
          int y = adjacent_hexes[i].y;
          if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
          if (damaged_zombies [x * BOARD_SIZE + y]) {
            printf("%d %d %d %d\n", xi, yi, x, y);
            zombies[x * BOARD_SIZE + y] = -1;
            zombies[xi * BOARD_SIZE + yi] = 2;
            damaged_zombies[x * BOARD_SIZE + y] = 0;
            goto ZOMBIE_END;
          }
        }

        // move zombies onto graveyards (if not there already)
        for (int i = 0; i < 6; i ++) {
          int x = adjacent_hexes[i].x;
          int y = adjacent_hexes[i].y;
          if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
          if (graveyards[x * BOARD_SIZE + y] 
              && zombies[x * BOARD_SIZE + y] == -1 
              && (enemy_captain.x != x || enemy_captain.y != y)
              && (own_captain.x != x || own_captain.y != y)) {
            zombies[xi * BOARD_SIZE + yi] = -1;
            zombies[x * BOARD_SIZE + y] = 2;
            printf("%d %d %d %d\n", xi, yi, x, y);
            goto ZOMBIE_END;
          }
        }

        // attack enemy zombies that can be hit with a second zombie
        for (int i = 0; i < 6; i ++) {
          int x = adjacent_hexes[i].x;
          int y = adjacent_hexes[i].y;
          if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
          if (zombies [x * BOARD_SIZE + y] == 1 - color) {
            location new_neighbors[6];
            get_adjacent_hexes(new_neighbors, adjacent_hexes + i);
            for (int j = 0; j < 6; j ++) {
              int nx = new_neighbors[j].x;
              int ny = new_neighbors[j].y;
              if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
              if (nx == xi && ny == yi) continue;
              if (zombies[nx * BOARD_SIZE + ny] == color) {
                printf("%d %d %d %d\n", xi, yi, x, y);
                damaged_zombies[x * BOARD_SIZE + y] = 1;
                zombies[xi * BOARD_SIZE + yi] = 2;
                goto ZOMBIE_END;
              }
            }
          }
        }

        // stay still if on a graveyard already
        if (graveyards[xi * BOARD_SIZE + yi])
          goto ZOMBIE_END;

        // move zombies randomly
        // Update: Don't move onto hexes adjacent to two enemy zombies
        for (int i = 0; i < 6; i ++) {
          int x = adjacent_hexes[i].x;
          int y = adjacent_hexes[i].y;
          if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
          if (zombies[x * BOARD_SIZE + y] == -1 
              && (enemy_captain.x != x || enemy_captain.y != y)
              && (own_captain.x != x || own_captain.y != y)) {
            // check new location to make sure it's safe
            location new_neighbors[6];
            get_adjacent_hexes(new_neighbors, adjacent_hexes + i);
            int adjacent_zombies = 0;
            for (int j = 0; j < 6; j ++) {
              int nx = new_neighbors[j].x;
              int ny = new_neighbors[j].y;
              if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
              if (zombies[nx * BOARD_SIZE + ny] == 1 - color) adjacent_zombies ++;
            }
            if (adjacent_zombies < 2) {
              printf("%d %d %d %d\n", xi, yi, x, y);
              zombies[xi * BOARD_SIZE + yi] = -1;
              zombies[x * BOARD_SIZE + y] = 2;
              goto ZOMBIE_END;
            }
          }
        }

        ZOMBIE_END:
          continue;
      }
    }

    // movement: charge enemy captain
    // update: if there is >= 1 unoccupied graveyard, charge nearest one instead
    int xi, yi, xf, yf, xe, ye, xt, yt, xc, yc;
    xi = own_captain.x;
    yi = own_captain.y;
    
    xt = xe = enemy_captain.x;
    yt = ye = enemy_captain.y;

    // look for unoccupied graveyard and set nearest one to target
    int min_dist = BOARD_SIZE * 2;
    for (int x = 0; x < BOARD_SIZE; x ++) {
      for (int y = 0; y < BOARD_SIZE; y ++) {
        if (graveyards[x * BOARD_SIZE + y] 
            && zombies[x * BOARD_SIZE + y] == -1
            && (x != xi || y != yi) && (x != xe || y != ye)) {
          int new_dist = dist(xi, yi, x, y);
          if (new_dist < min_dist) {
            min_dist = new_dist;
            xt = x;
            yt = y;
          }
        }
      }
    }
    
    xf = xi;
    yf = yi;

    xc = (BOARD_SIZE - 1)/2;
    yc = (BOARD_SIZE - 1)/2;

    int distance = 10 * dist(xi, yi, xt, yt) + dist(xi, yi, xc, yc);
    // look for empty adjacent hex with smaller distance
    get_adjacent_hexes(adjacent_hexes, &own_captain);
    for (int i = 0; i < 6; i ++) {
      int x = adjacent_hexes[i].x;
      int y = adjacent_hexes[i].y;
      if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
      if (zombies [x * BOARD_SIZE + y] != -1 && zombies [x * BOARD_SIZE + y] != color) continue;
      if ((xe == x) && (ye == y)) continue;
      int new_dist = 10 * dist(x, y, xt, yt) + dist(x, y, xc, yc);
      if (new_dist < distance) {
        distance = new_dist;
        xf = x;
        yf = y;
      }
    }
    if (xi != xf || yi != yf)
      printf("%d %d %d %d\n", xi, yi, xf, yf);
    location new_loc;
    new_loc.x = xf;
    new_loc.y = yf;

    // attack enemy zombies with captain
    get_adjacent_hexes(adjacent_hexes, &new_loc);
    for (int i = 0; i < 6; i ++) {
      int x = adjacent_hexes[i].x;
      int y = adjacent_hexes[i].y;
      if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
      if (zombies [x * BOARD_SIZE + y] == 1 - color) {
        zombies[x * BOARD_SIZE + y] = -1;
        printf("%d %d %d %d\n", xf, yf, x, y);
        break;
      }
    }

    // end move phase
    printf("\n");
    fflush(stdout);

    // spawn zombies on adjacent graveyards
    get_adjacent_hexes(adjacent_hexes, &new_loc);
    for (int i = 0; i < 6; i ++) {
      if (money[color] < 2) break;
      int x = adjacent_hexes[i].x;
      int y = adjacent_hexes[i].y;
      if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
      if (graveyards[x * BOARD_SIZE + y] && zombies[x * BOARD_SIZE + y] == -1) {
        printf("1 %d %d\n", x, y);
        zombies[x * BOARD_SIZE + y] = color;
        money[color] -= 2;
      }
    }
    // spawn zombies on other adjacent locations
    // TODO: More elegant to check if there's actually money first
    int best_x, best_y;
    for (int j = 0; j < 6; j++) {
      if (money[color] < 2) break;
      best_x = -1;
      int best_dist = BOARD_SIZE * 2;
      for (int i = 0; i < 6; i ++) {
        int x = adjacent_hexes[i].x;
        int y = adjacent_hexes[i].y;
        if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
        if (zombies[x * BOARD_SIZE + y] != -1) continue;
        if (x == xe && y == ye) continue;
        // check new location to make sure it's safe
        location new_neighbors[6];
        get_adjacent_hexes(new_neighbors, adjacent_hexes + i);
        int adjacent_zombies = 0;
        for (int j = 0; j < 6; j ++) {
          int nx = new_neighbors[j].x;
          int ny = new_neighbors[j].y;
          if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
          if (zombies[nx * BOARD_SIZE + ny] == 1 - color) adjacent_zombies ++;
        }
        if (adjacent_zombies >= 2) continue;
        int new_dist = dist(x, y, xe, ye);
        if (new_dist >= best_dist) continue;
        best_dist = new_dist;
        best_x = x;
        best_y = y;
      }
      if (best_x == -1) break;
      printf("1 %d %d\n", best_x, best_y);
      zombies[best_x * BOARD_SIZE + best_y] = color;
      money[color] -= 2;
    }
    printf("\n");
    fflush(stdout);
  }
}
