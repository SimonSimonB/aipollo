const POINT_RADIUS = 1
const COLORS = { 'half_note': 'red', 'quarter_note': 'green' }

function init(scoreElementsJson) {
    scoreElements = renderScoreElements(scoreElementsJson);
    var totalPositionChange = new WeakMap()
    for (const scoreElementSvg of scoreElements) {
        totalPositionChange.set(scoreElementSvg, { x: 0, y: 0 })
    }

    function renderScoreElements(scoreElementsJsonParsed) {
        var scoreElements = []
        for (const scoreElement of scoreElementsJsonParsed) {
            elementType = scoreElement.type._name_

            // Don't draw staff lines.
            if (elementType == 'staff') {
                continue;
            }
            var scoreElementSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            scoreElementSvg.classList.add('draggable');
            scoreElementSvg.classList.add('scoresymbol');

            const minY = Math.min(...scoreElement.pixels.map(p => p.y));
            const minX = Math.min(...scoreElement.pixels.map(p => p.x));
            const maxY = Math.max(...scoreElement.pixels.map(p => p.y));
            const maxX = Math.max(...scoreElement.pixels.map(p => p.x));

            scoreElementSvg.setAttribute('height', maxY - minY);
            scoreElementSvg.setAttribute('width', maxX - minX);
            scoreElementSvg.style.top = minY;
            scoreElementSvg.style.left = minX;

            // Create a SVG circle for each pixel belonging to the score element.
            for (const pixel of scoreElement.pixels) {
                var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cy', pixel.y - minY);
                circle.setAttribute('cx', pixel.x - minX);
                circle.setAttribute('r', POINT_RADIUS);
                scoreElementSvg.appendChild(circle)

                circle.style.fill = COLORS[elementType]
                circle.style.fillOpacity = 0.3
            }

            scoreElements.push(scoreElementSvg)
            scoreElements.concat(renderScoreElements(scoreElement.children))
            document.getElementById('imagediv').appendChild(scoreElementSvg);
        }

        return scoreElements
    }

    interact('.draggable').draggable({
        listeners: {
            start(event) {
                console.log(event.type, event.target)
            },
            move(event) {
                movedId = event.target;
                totalPositionChange.get(movedId).x += event.dx;
                totalPositionChange.get(movedId).y += event.dy;

                event.target.style.transform =
                    `translate(${totalPositionChange.get(movedId).x}px, ${totalPositionChange.get(movedId).y}px)`
            },
        }
    })
}

