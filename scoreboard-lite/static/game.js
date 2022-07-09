// Load game state
var game_state = JSON.parse(document.getElementById("game_state").value);
var game_reset_state = JSON.parse(document.getElementById("game_reset_state").value);
const dimension = game_state.board.length;

//
// Coordinate Conversions
//

function getUICoordinates(i, j) {
    const row = i + j;
    const col = (row % 2 != dimension % 2 ? (i - j + dimension - 1) / 2 : (i - j + dimension - 2) / 2);
    return [row, col];
}

function getCodeCoordinates(row, col) {
    const i = (row % 2 != dimension % 2 ? (2 * col + row - dimension + 1) / 2 : (2 * col + row - dimension + 2) / 2);
    const j = row - i;
    return [i, j];
}

//
// UI Setup
//

function clickButton(i, j, button) {
    // Display coordinates
    document.getElementById("active").querySelectorAll("div.hex-row")[0].lastChild.children[1].innerText = i + ", " + j;

    // Left click on unit
    if (button == 0 && game_state.board[i][j].unit != null) {
        document.getElementsByName("move_from_i")[0].value = i;
        document.getElementsByName("move_from_j")[0].value = j;
    }
    if (button == 2) {  // Right click
        document.getElementsByName("move_to_i")[0].value = i;
        document.getElementsByName("move_to_j")[0].value = j;

        // Submit the form with a spoofed "move" action
        var move = document.createElement("input");
        move.value = "move";
        move.setAttribute("type", "hidden");
        move.setAttribute("name", "move");
        document.getElementById("the_form").appendChild(move);
        document.getElementById("the_form").submit();
    }
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
function makeHexGrid(board) {
    var game = board.id == "active" ? game_state : game_reset_state;
    var interactive = board.id == "active";
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
                const hex = game.board[i][j];
                var space = makeSpace("hex")
                if (hex.is_graveyard) {
                    space.setAttribute("class", "hex graveyard");
                } else if (hex.is_water) {
                    space.setAttribute("class", "hex water");
                }

                if (hex.unit != null) {
                    var buttonWrapper = document.createElement("button");
                    var unitButton = document.createElement("div");
                    if (interactive && hex.unit.color == 0) {
                        var attackDiv = document.createElement("div");
                        if (game.phase == "move" && hex.unit.remainingAttack > 0) {
                            attackDiv.innerText = "⚔️";
                        }
                        attackDiv.setAttribute("class", "stat");
                        unitButton.appendChild(attackDiv);
                    }
                    var nameDiv = document.createElement("div");
                    nameDiv.innerText = hex.unit.type[0];
                    nameDiv.setAttribute("class", "unit-name");
                    unitButton.appendChild(nameDiv);
                    if (interactive && hex.unit.color == 0) {
                        var moveDiv = document.createElement("div");
                        if (game.phase == "move" && !hex.unit.hasMoved) {
                            moveDiv.innerText = "➜";
                        }
                        moveDiv.setAttribute("class", "stat");
                        unitButton.appendChild(moveDiv);
                    }
                    unitButton.setAttribute("class", "unit-button color" + hex.unit.color);
                    buttonWrapper.appendChild(unitButton);
                    space.children[1].appendChild(buttonWrapper);
                }
                if (interactive) {
                    space.children[1].addEventListener('mouseup', function (e) {
                        clickButton(i, j, e.button);
                    }, false);
                }
                row.appendChild(space);
            }
        }
        board.appendChild(row);
    }

    // Money display
    var rows = board.querySelectorAll("div.hex-row");
    rows[0].querySelectorAll(".hex")[0].children[1].innerText = "$" + game.money[0]
    rows[0].querySelectorAll(".hex")[0].children[1].setAttribute("class", "middle color0")
    rows[rows.length - 1].querySelectorAll(".hex")[0].children[1].innerText = "$" + game.money[1]
    rows[rows.length - 1].querySelectorAll(".hex")[0].children[1].setAttribute("class", "middle color1")
}


var games = document.getElementsByName("game");
games.forEach(makeHexGrid);