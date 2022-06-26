// Load game state
var game_state = JSON.parse(document.getElementById("game_state").value);
const dimension = game_state.board.length;

//
// Coordinate Conversions
//

function getUICoordinates(i, j) {
    const row = i+j;
    const col = (row % 2 != dimension % 2 ? (i - j + dimension - 1) / 2 : (i - j + dimension - 2) / 2);
    return [row, col];
}

function getCodeCoordinates(row, col) {
    const i = (row % 2 != dimension % 2 ? (2*col + row - dimension + 1) / 2 : (2*col + row - dimension + 2) / 2);
    const j = row - i;
    return [i, j];
}

//
// UI Setup
//

// Click any square displays its coordinates
function clickButton(i, j) {
    document.getElementById("game").firstChild.lastChild.children[1].innerText = i + ", " + j;
}

// Helper to make one hex
function makeSpace(classname) {
    var space = document.createElement("div");
    space.setAttribute("class", classname);
    var top_part = document.createElement("div");
    top_part.setAttribute("class", "top");
    space.appendChild(top_part);
    var middle = document.createElement("div");
    middle.setAttribute("class", "middle");
    space.appendChild(middle);
    var bottom = document.createElement("div");
    bottom.setAttribute("class", "bottom");
    space.appendChild(bottom);
    return space;
}

// Hex grid
var game = document.getElementById("game");
for (let y = 0; y < 2 * dimension - 1; y++) {
    var row = document.createElement("div");
    if (y % 2 == dimension % 2) {  // offset rows such that widest row gets no offset
        row.setAttribute("class", "hex-row offset");
    } else {
        row.setAttribute("class", "hex-row");
    }
    const numPlayableInRow = (y < dimension ? y + 1 : 2 * dimension - 1 - y);
    const totalInRow = (y % 2 != dimension % 2 ? dimension : dimension - 1);
    for (let x = 0; x < totalInRow; x++) {
        if (x < (totalInRow - numPlayableInRow) / 2 || x >= (totalInRow + numPlayableInRow) / 2) {
            row.appendChild(makeSpace("hex transparent"));
        } else {
            const [i, j] = getCodeCoordinates(y, x);
            const hex = game_state.board[i][j];
            var space = makeSpace("hex")
            if (hex.is_graveyard) {
                space.setAttribute("class", "hex graveyard");
            } else if (hex.is_water) {
                space.setAttribute("class", "hex water");
            }

            if (hex.unit != null) {
                var unitButton = document.createElement("button");
                unitButton.innerText = hex.unit.type[0];
                unitButton.setAttribute("class", "color"+hex.unit.color)
                space.children[1].appendChild(unitButton);
            }
            space.children[1].addEventListener('click', function() {
                clickButton(i, j);
            }, false);
            row.appendChild(space);
        }
    }
    game.appendChild(row);
}

// Money display
game.firstChild.firstChild.children[1].innerText="$"+game_state.money[0]
game.firstChild.firstChild.children[1].setAttribute("class", "middle color0")
game.lastChild.firstChild.children[1].innerText="$"+game_state.money[1]
game.lastChild.firstChild.children[1].setAttribute("class", "middle color1")