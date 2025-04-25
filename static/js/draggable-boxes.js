/**
 * Draggable Target Boxes for Elliott Wave Analysis
 * 
 * This script adds drag functionality to the target boxes in the chart.
 * It works by intercepting the Plotly chart and adding custom event handlers.
 */

// Store the original positions of boxes for reset functionality
let originalBoxPositions = {};
let activeBoxId = null;
let isDragging = false;
let dragStartY = 0;
let currentBox = null;
let boxesData = [];

// Initialize the draggable functionality
function initDraggableBoxes() {
    const plotDiv = document.querySelector('.js-plotly-plot');
    if (!plotDiv) return;

    // Wait for Plotly to be fully loaded
    if (typeof Plotly === 'undefined') {
        setTimeout(initDraggableBoxes, 500);
        return;
    }

    // Extract all target boxes from the plot
    const gd = plotDiv._fullLayout;
    if (!gd || !gd.shapes) return;

    // Find all rectangle shapes (target boxes)
    boxesData = gd.shapes.filter(shape => shape.type === 'rect');
    
    // Store original positions
    boxesData.forEach((box, index) => {
        originalBoxPositions[index] = {
            y0: box.y0,
            y1: box.y1
        };
    });

    // Add event listeners for dragging
    plotDiv.addEventListener('mousedown', startDrag);
    document.addEventListener('mousemove', dragBox);
    document.addEventListener('mouseup', endDrag);

    // Add reset button
    addResetButton(plotDiv);
    
    // Add info message about draggable functionality
    addInfoMessage();
}

// Start dragging a box
function startDrag(e) {
    const plotDiv = document.querySelector('.js-plotly-plot');
    if (!plotDiv) return;
    
    const rect = plotDiv.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Get the Plotly data
    const gd = plotDiv._fullLayout;
    if (!gd || !gd.shapes) return;
    
    // Check if click is inside any box
    for (let i = 0; i < boxesData.length; i++) {
        const box = boxesData[i];
        const xInBox = x >= gd._xaxis.l2p(box.x0) && x <= gd._xaxis.l2p(box.x1);
        const yInBox = y >= gd._yaxis.l2p(box.y1) && y <= gd._yaxis.l2p(box.y0);
        
        if (xInBox && yInBox) {
            activeBoxId = i;
            isDragging = true;
            dragStartY = y;
            currentBox = box;
            
            // Change cursor to indicate dragging
            document.body.style.cursor = 'ns-resize';
            
            // Highlight the active box
            Plotly.relayout(plotDiv, {
                [`shapes[${i}].line.width`]: 3,
                [`shapes[${i}].line.dash`]: 'solid'
            });
            
            break;
        }
    }
}

// Drag the box
function dragBox(e) {
    if (!isDragging || activeBoxId === null) return;
    
    const plotDiv = document.querySelector('.js-plotly-plot');
    if (!plotDiv) return;
    
    const rect = plotDiv.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const deltaY = y - dragStartY;
    
    // Convert pixel movement to data coordinates
    const gd = plotDiv._fullLayout;
    const yAxis = gd._yaxis;
    
    if (!yAxis) return;
    
    // Calculate new positions
    const pixelToDataRatio = Math.abs(yAxis.l2p(currentBox.y0) - yAxis.l2p(currentBox.y1)) / 
                            Math.abs(currentBox.y0 - currentBox.y1);
    
    const dataShift = deltaY / pixelToDataRatio;
    
    // Update the box position
    const updates = {
        [`shapes[${activeBoxId}].y0`]: currentBox.y0 - dataShift,
        [`shapes[${activeBoxId}].y1`]: currentBox.y1 - dataShift
    };
    
    // Update the annotation position if it exists
    const annotations = gd.annotations || [];
    for (let i = 0; i < annotations.length; i++) {
        const ann = annotations[i];
        // Check if this annotation is associated with our box
        if (ann.x >= currentBox.x0 && ann.x <= currentBox.x1 && 
            ann.y >= currentBox.y1 && ann.y <= currentBox.y0) {
            updates[`annotations[${i}].y`] = ann.y - dataShift;
        }
    }
    
    Plotly.relayout(plotDiv, updates);
    dragStartY = y;
}

// End dragging
function endDrag() {
    if (!isDragging) return;
    
    isDragging = false;
    document.body.style.cursor = 'default';
    
    const plotDiv = document.querySelector('.js-plotly-plot');
    if (!plotDiv || activeBoxId === null) return;
    
    // Reset line style
    Plotly.relayout(plotDiv, {
        [`shapes[${activeBoxId}].line.width`]: 1,
        [`shapes[${activeBoxId}].line.dash`]: boxesData[activeBoxId].line.dash || 'dash'
    });
    
    // Update the box data
    const gd = plotDiv._fullLayout;
    if (gd && gd.shapes && gd.shapes[activeBoxId]) {
        boxesData[activeBoxId] = gd.shapes[activeBoxId];
    }
    
    activeBoxId = null;
    currentBox = null;
    
    // Update the price values in annotations
    updateAnnotationPrices(plotDiv);
}

// Update price values in annotations after dragging
function updateAnnotationPrices(plotDiv) {
    const gd = plotDiv._fullLayout;
    if (!gd || !gd.shapes || !gd.annotations) return;
    
    for (let i = 0; i < gd.shapes.length; i++) {
        const box = gd.shapes[i];
        
        // Find associated annotation
        for (let j = 0; j < gd.annotations.length; j++) {
            const ann = gd.annotations[j];
            
            // Check if this annotation is associated with our box
            if (ann.x >= box.x0 && ann.x <= box.x1 && 
                ann.y >= box.y1 && ann.y <= box.y0) {
                
                // Extract the text and update price range
                let text = ann.text;
                const priceRangeRegex = /(\d+\.\d+)(?:\s*\([+-]?\d+\.\d+%\))?\s*-\s*(\d+\.\d+)(?:\s*\([+-]?\d+\.\d+%\))?/;
                const match = text.match(priceRangeRegex);
                
                if (match) {
                    // Format new price values
                    const newLow = box.y0 < box.y1 ? box.y0 : box.y1;
                    const newHigh = box.y0 > box.y1 ? box.y0 : box.y1;
                    const newPriceText = `${newLow.toFixed(2)} - ${newHigh.toFixed(2)}`;
                    
                    // Replace the old price range with the new one
                    text = text.replace(priceRangeRegex, newPriceText);
                    
                    // Update the annotation
                    Plotly.relayout(plotDiv, {
                        [`annotations[${j}].text`]: text
                    });
                }
            }
        }
    }
}

// Add a reset button to restore original positions
function addResetButton(plotDiv) {
    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'Reset Target Boxes';
    resetBtn.className = 'reset-boxes-btn';
    resetBtn.style.position = 'absolute';
    resetBtn.style.top = '10px';
    resetBtn.style.right = '10px';
    resetBtn.style.zIndex = '999';
    resetBtn.style.padding = '8px 12px';
    resetBtn.style.backgroundColor = '#4d84ff';
    resetBtn.style.color = 'white';
    resetBtn.style.border = 'none';
    resetBtn.style.borderRadius = '4px';
    resetBtn.style.cursor = 'pointer';
    resetBtn.style.fontWeight = '500';
    resetBtn.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    
    resetBtn.addEventListener('click', () => {
        resetBoxPositions(plotDiv);
    });
    
    // Add the button to the plot container
    plotDiv.parentNode.style.position = 'relative';
    plotDiv.parentNode.appendChild(resetBtn);
}

// Reset all boxes to their original positions
function resetBoxPositions(plotDiv) {
    const updates = {};
    
    Object.keys(originalBoxPositions).forEach(index => {
        updates[`shapes[${index}].y0`] = originalBoxPositions[index].y0;
        updates[`shapes[${index}].y1`] = originalBoxPositions[index].y1;
    });
    
    Plotly.relayout(plotDiv, updates);
    
    // Update the annotations as well
    setTimeout(() => {
        updateAnnotationPrices(plotDiv);
    }, 100);
}

// Add info message about draggable functionality
function addInfoMessage() {
    const infoDiv = document.createElement('div');
    infoDiv.className = 'draggable-info';
    infoDiv.innerHTML = '<strong>✓ Draggable Target Boxes Enabled</strong> - Click and drag any target box to adjust its position. Use the Reset button to restore original positions.';
    infoDiv.style.backgroundColor = 'rgba(77, 132, 255, 0.1)';
    infoDiv.style.color = '#4d84ff';
    infoDiv.style.padding = '10px 15px';
    infoDiv.style.borderRadius = '4px';
    infoDiv.style.marginBottom = '15px';
    infoDiv.style.border = '1px solid rgba(77, 132, 255, 0.3)';
    infoDiv.style.fontSize = '0.9em';
    
    const plotContainer = document.querySelector('.plot-container');
    if (plotContainer && plotContainer.parentNode) {
        plotContainer.parentNode.insertBefore(infoDiv, plotContainer);
    }
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for Plotly to render the chart
    setTimeout(initDraggableBoxes, 1000);
});

// Re-initialize when Plotly updates the chart
document.addEventListener('plotly_afterplot', function() {
    setTimeout(initDraggableBoxes, 500);
});
