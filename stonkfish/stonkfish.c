#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#define BOARD_SIZE 5

// timeout (in sec)
#define TIMEOUT 10

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

void clear_zombies(int * board) {
  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i ++)
    board[i] = -1;
}

void clear_graveyards(int * board) {
  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i ++)
    board[i] = 0;
}

int main() {
  char line [256];
  int color;
  location own_captain;
  location enemy_captain;
  location adjacent_hexes[6];
  int graveyards [BOARD_SIZE * BOARD_SIZE];
  int water [BOARD_SIZE * BOARD_SIZE] = {0};

  // zombies on board (-1 for empty, 0/1 for color)
  int zombies [BOARD_SIZE * BOARD_SIZE];

  // timer (to determine when Python parent function has died)
  signal(SIGALRM, sig_handler);

  FILE * log;
  log = fopen("log.txt", "w");
  while (1) {
    // reset alarm
    alarm(TIMEOUT);

    // reset graveyards and zombies
    clear_graveyards(graveyards);
    clear_zombies(zombies);

    line[0] = '\0';
    while (strcmp(line, "Your turn\n") != 0) {
      usleep(10); // sleep for 0.1 ms
      char * captain = " N ";
      char * zombie = " Z ";
      char * graveyard = " True\n";
      fgets(line, 256, stdin);
      //fprintf(log, "%s", line);
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

        // move zombies onto graveyards (if not there already)
        for (int i = 0; i < 6; i ++) {
          int x = adjacent_hexes[i].x;
          int y = adjacent_hexes[i].y;
          if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
          if (graveyards[x * BOARD_SIZE + y] 
              && zombies[x * BOARD_SIZE + y] == -1 
              && (enemy_captain.x != x || enemy_captain.y != y)
              && (own_captain.x != x || own_captain.y != y)) {
            printf("%d %d %d %d\n", xi, yi, x, y);
            goto ZOMBIE_END;
          }
        }

        // attack enemy zombies with other zombies
        for (int i = 0; i < 6; i ++) {
          int x = adjacent_hexes[i].x;
          int y = adjacent_hexes[i].y;
          if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
          if (zombies [x * BOARD_SIZE + y] == 1 - color) {
            printf("%d %d %d %d\n", xi, yi, x, y);
            goto ZOMBIE_END;
          }
        }

        // stay still if on a graveyard already
        if (graveyards[xi * BOARD_SIZE + yi])
          goto ZOMBIE_END;

        // move zombies randomly
        for (int i = 0; i < 6; i ++) {
          int x = adjacent_hexes[i].x;
          int y = adjacent_hexes[i].y;
          if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
          if (zombies[x * BOARD_SIZE + y] == -1 
              && (enemy_captain.x != x || enemy_captain.y != y)
              && (own_captain.x != x || own_captain.y != y)) {
            printf("%d %d %d %d\n", xi, yi, x, y);
            goto ZOMBIE_END;
          }
        }

        ZOMBIE_END:
          continue;
      }
    }

    // movement: charge enemy captain
    int xi, yi, xf, yf;
    xi = own_captain.x;
    yi = own_captain.y;
    xf = xi;
    yf = yi;
    if (abs(yf - enemy_captain.y) >= abs(xf - enemy_captain.x))
      yf -= sign(own_captain.y - enemy_captain.y);
    else
      xf -= sign(own_captain.x - enemy_captain.x);
    location new_loc;
    if (zombies[xf * BOARD_SIZE + yf] != (1-color) && (enemy_captain.x != xf || enemy_captain.y != yf)) {
      printf("%d %d %d %d\n", xi, yi, xf, yf);
    } else {
      xf = xi;
      yf = yi;
    }
    new_loc.x = xf;
    new_loc.y = yf;

    // attack enemy zombies with captain
    get_adjacent_hexes(adjacent_hexes, &new_loc);
    for (int i = 0; i < 6; i ++) {
      int x = adjacent_hexes[i].x;
      int y = adjacent_hexes[i].y;
      if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
      if (zombies [x * BOARD_SIZE + y] == 1 - color) {
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
      int x = adjacent_hexes[i].x;
      int y = adjacent_hexes[i].y;
      if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
      if (graveyards[x * BOARD_SIZE + y])
        printf("1 %d %d\n", x, y);
    }
    // spawn zombies on other adjacent locations
    // TODO: More elegant to check if there's actually money first
    for (int i = 0; i < 6; i ++) {
      int x = adjacent_hexes[i].x;
      int y = adjacent_hexes[i].y;
      if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;
      printf("1 %d %d\n", x, y);
    }
    printf("\n");
    fflush(stdout);
  }
}
