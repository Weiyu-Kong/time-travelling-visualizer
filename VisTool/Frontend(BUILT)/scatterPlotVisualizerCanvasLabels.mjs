/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import * as d3 from 'd3';
import * as THREE from 'three';
import { CameraType } from './renderContext';
import { CollisionGrid } from './label';
import * as util from './util';
const MAX_LABELS_ON_SCREEN = 10000;
const LABEL_STROKE_WIDTH = 3;
const LABEL_FILL_WIDTH = 6;
/**
 * Creates and maintains a 2d canvas on top of the GL canvas. All labels, when
 * active, are rendered to the 2d canvas as part of the visible render pass.
 */
export class ScatterPlotVisualizerCanvasLabels {
    constructor(container) {
        this.labelsActive = true;
        this.canvas = document.createElement('canvas');
        container.appendChild(this.canvas);
        this.gc = this.canvas.getContext('2d');
        this.canvas.style.position = 'absolute';
        this.canvas.style.left = '0';
        this.canvas.style.top = '0';
        this.canvas.style.pointerEvents = 'none';
    }
    removeAllLabels() {
        const pixelWidth = this.canvas.width * window.devicePixelRatio;
        const pixelHeight = this.canvas.height * window.devicePixelRatio;
        this.gc.clearRect(0, 0, pixelWidth, pixelHeight);
    }
    /** Render all of the non-overlapping visible labels to the canvas. */
    makeLabels(rc) {
        if (rc.labels == null || rc.labels.pointIndices.length === 0) {
            return;
        }
        if (this.worldSpacePointPositions == null) {
            return;
        }
        const lrc = rc.labels;
        const sceneIs3D = rc.cameraType === CameraType.Perspective;
        const labelHeight = parseInt(this.gc.font, 10);
        const dpr = window.devicePixelRatio;
        let grid;
        {
            const pixw = this.canvas.width * dpr;
            const pixh = this.canvas.height * dpr;
            const bb = { loX: 0, hiX: pixw, loY: 0, hiY: pixh };
            grid = new CollisionGrid(bb, pixw / 25, pixh / 50);
        }
        let opacityMap = d3
            .scalePow()
            .exponent(Math.E)
            .domain([rc.farthestCameraSpacePointZ, rc.nearestCameraSpacePointZ])
            .range([0.1, 1]);
        const camPos = rc.camera.position;
        const camToTarget = camPos.clone().sub(rc.cameraTarget);
        let camToPoint = new THREE.Vector3();
        this.gc.textBaseline = 'middle';
        this.gc.miterLimit = 2;
        // Have extra space between neighboring labels. Don't pack too tightly.
        const labelMargin = 0;
        // Shift the label to the right of the point circle.
        const xShift = 4;
        const n = Math.min(MAX_LABELS_ON_SCREEN, lrc.pointIndices.length);
        for (let i = 0; i < n; ++i) {
            let point;
            {
                const pi = lrc.pointIndices[i];
                point = util.vector3FromPackedArray(this.worldSpacePointPositions, pi);
            }
            // discard points that are behind the camera
            camToPoint.copy(camPos).sub(point);
            if (camToTarget.dot(camToPoint) < 0) {
                continue;
            }
            let [x, y] = util.vector3DToScreenCoords(rc.camera, rc.screenWidth, rc.screenHeight, point);
            x += xShift;
            // Computing the width of the font is expensive,
            // so we assume width of 1 at first. Then, if the label doesn't
            // conflict with other labels, we measure the actual width.
            const textBoundingBox = {
                loX: x - labelMargin,
                hiX: x + 1 + labelMargin,
                loY: y - labelHeight / 2 - labelMargin,
                hiY: y + labelHeight / 2 + labelMargin,
            };
            if (grid.insert(textBoundingBox, true)) {
                const text = lrc.labelStrings[i];
                const fontSize = lrc.defaultFontSize * lrc.scaleFactors[i] * dpr;
                this.gc.font = fontSize + 'px roboto';
                // Now, check with properly computed width.
                textBoundingBox.hiX += this.gc.measureText(text).width - 1;
                if (grid.insert(textBoundingBox)) {
                    let opacity = 1;
                    if (sceneIs3D && lrc.useSceneOpacityFlags[i] === 1) {
                        opacity = opacityMap(camToPoint.length());
                    }
                    this.gc.fillStyle = this.styleStringFromPackedRgba(lrc.fillColors, i, opacity);
                    this.gc.strokeStyle = this.styleStringFromPackedRgba(lrc.strokeColors, i, opacity);
                    this.gc.lineWidth = LABEL_STROKE_WIDTH;
                    this.gc.strokeText(text, x, y);
                    this.gc.lineWidth = LABEL_FILL_WIDTH;
                    this.gc.fillText(text, x, y);
                }
            }
        }
    }
    styleStringFromPackedRgba(packedRgbaArray, colorIndex, opacity) {
        const offset = colorIndex * 3;
        const r = packedRgbaArray[offset];
        const g = packedRgbaArray[offset + 1];
        const b = packedRgbaArray[offset + 2];
        return 'rgba(' + r + ',' + g + ',' + b + ',' + opacity + ')';
    }
    onResize(newWidth, newHeight) {
        let dpr = window.devicePixelRatio;
        this.canvas.width = newWidth * dpr;
        this.canvas.height = newHeight * dpr;
        this.canvas.style.width = newWidth + 'px';
        this.canvas.style.height = newHeight + 'px';
    }
    dispose() {
        this.removeAllLabels();
        this.canvas = null;
        this.gc = null;
    }
    onPointPositionsChanged(newPositions) {
        this.worldSpacePointPositions = newPositions;
        this.removeAllLabels();
    }
    onRender(rc) {
        if (!this.labelsActive) {
            return;
        }
        this.removeAllLabels();
        this.makeLabels(rc);
    }
    setScene(scene) { }
    onPickingRender(renderContext) { }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2NhdHRlclBsb3RWaXN1YWxpemVyQ2FudmFzTGFiZWxzLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGVuc29yYm9hcmQvcHJvamVjdG9yL3NjYXR0ZXJQbG90VmlzdWFsaXplckNhbnZhc0xhYmVscy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7OztnRkFhZ0Y7QUFDaEYsT0FBTyxLQUFLLEVBQUUsTUFBTSxJQUFJLENBQUM7QUFDekIsT0FBTyxLQUFLLEtBQUssTUFBTSxPQUFPLENBQUM7QUFFL0IsT0FBTyxFQUFDLFVBQVUsRUFBZ0IsTUFBTSxpQkFBaUIsQ0FBQztBQUMxRCxPQUFPLEVBQWMsYUFBYSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sUUFBUSxDQUFDO0FBRS9CLE1BQU0sb0JBQW9CLEdBQUcsS0FBSyxDQUFDO0FBQ25DLE1BQU0sa0JBQWtCLEdBQUcsQ0FBQyxDQUFDO0FBQzdCLE1BQU0sZ0JBQWdCLEdBQUcsQ0FBQyxDQUFDO0FBQzNCOzs7R0FHRztBQUNILE1BQU0sT0FBTyxpQ0FBaUM7SUFNNUMsWUFBWSxTQUFzQjtRQUQxQixpQkFBWSxHQUFZLElBQUksQ0FBQztRQUVuQyxJQUFJLENBQUMsTUFBTSxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUMsUUFBUSxDQUFDLENBQUM7UUFDL0MsU0FBUyxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDbkMsSUFBSSxDQUFDLEVBQUUsR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUN2QyxJQUFJLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxRQUFRLEdBQUcsVUFBVSxDQUFDO1FBQ3hDLElBQUksQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLElBQUksR0FBRyxHQUFHLENBQUM7UUFDN0IsSUFBSSxDQUFDLE1BQU0sQ0FB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Q0FBQyxlQUFlLEVBQUUsQ0FBQztJQUN6QixDQUFDO0lBQ0QsUUFBUSxDQUFDLEVBQWlCO1FBQ3hCLElBQUksQ0FBQyxJQUFJLENBQUMsWUFBWSxFQUFFO1lBQ3RCLE9BQU87U0FDUjtRQUNELElBQUksQ0FBQyxlQUFlLEVBQUUsQ0FBQztRQUN2QixJQUFJLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxDQUFDO0lBQ3RCLENBQUM7SUFDRCxRQUFRLENBQUMsS0FBa0IsSUFBRyxDQUFDO0lBQy9CLGVBQWUsQ0FBQyxhQUE0QixJQUFHLENBQUM7Q0FDakQiLCJzb3VyY2VzQ29udGVudCI6WyIvKiBDb3B5cmlnaHQgMjAxNiBUaGUgVGVuc29yRmxvdyBBdXRob3JzLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5pbXBvcnQgKiBhcyBkMyBmcm9tICdkMyc7XG5pbXBvcnQgKiBhcyBUSFJFRSBmcm9tICd0aHJlZSc7XG5cbmltcG9ydCB7Q2FtZXJhVHlwZSwgUmVuZGVyQ29udGV4dH0gZnJvbSAnLi9yZW5kZXJDb250ZXh0JztcbmltcG9ydCB7Qm91bmRpbmdCb3gsIENvbGxpc2lvbkdyaWR9IGZyb20gJy4vbGFiZWwnO1xuaW1wb3J0IHtTY2F0dGVyUGxvdFZpc3VhbGl6ZXJ9IGZyb20gJy4vc2NhdHRlclBsb3RWaXN1YWxpemVyJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi91dGlsJztcblxuY29uc3QgTUFYX0xBQkVMU19PTl9TQ1JFRU4gPSAxMDAwMDtcbmNvbnN0IExBQkVMX1NUUk9LRV9XSURUSCA9IDM7XG5jb25zdCBMQUJFTF9GSUxMX1dJRFRIID0gNjtcbi8qKlxuICogQ3JlYXRlcyBhbmQgbWFpbnRhaW5zIGEgMmQgY2FudmFzIG9uIHRvcCBvZiB0aGUgR0wgY2FudmFzLiBBbGwgbGFiZWxzLCB3aGVuXG4gKiBhY3RpdmUsIGFyZSByZW5kZXJlZCB0byB0aGUgMmQgY2FudmFzIGFzIHBhcnQgb2YgdGhlIHZpc2libGUgcmVuZGVyIHBhc3MuXG4gKi9cbmV4cG9ydCBjbGFzcyBTY2F0dGVyUGxvdFZpc3VhbGl6ZXJDYW52YXNMYWJlbHNcbiAgaW1wbGVtZW50cyBTY2F0dGVyUGxvdFZpc3VhbGl6ZXIge1xuICBwcml2YXRlIHdvcmxkU3BhY2VQb2ludFBvc2l0aW9uczogRmxvYXQzMkFycmF5O1xuICBwcml2YXRlIGdjOiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG4gIHByaXZhdGUgY2FudmFzOiBIVE1MQ2FudmFzRWxlbWVudDtcbiAgcHJpdmF0ZSBsYWJlbHNBY3RpdmU6IGJvb2xlYW4gPSB0cnVlO1xuICBjb25zdHJ1Y3Rvcihjb250YWluZXI6IEhUTUxFbGVtZW50KSB7XG4gICAgdGhpcy5jYW52YXMgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdjYW52YXMnKTtcbiAgICBjb250YWluZXIuYXBwZW5kQ2hpbGQodGhpcy5jYW52YXMpO1xuICAgIHRoaXMuZ2MgPSB0aGlzLmNhbnZhcy5nZXRDb250ZXh0KCcyZCcpO1xuICAgIHRoaXMuY2FudmFzLnN0eWxlLnBvc2l0aW9uID0gJ2Fic29sdXRlJztcbiAgICB0aGlzLmNhbnZhcy5zdHlsZS5sZWZ0ID0gJzAnO1xuICAgIHRoaXMuY2FudmFzLnN0eWxlLnRvcCA9ICcwJztcbiAgICB0aGlzLmNhbnZhcy5zdHlsZS5wb2ludGVyRXZlbnRzID0gJ25vbmUnO1xuICB9XG4gIHByaXZhdGUgcmVtb3ZlQWxsTGFiZWxzKCkge1xuICAgIGNvbnN0IHBpeGVsV2lkdGggPSB0aGlzLmNhbnZhcy53aWR0aCAqIHdpbmRvdy5kZXZpY2VQaXhlbFJhdGlvO1xuICAgIGNvbnN0IHBpeGVsSGVpZ2h0ID0gdGhpcy5jYW52YXMuaGVpZ2h0ICogd2luZG93LmRldmljZVBpeGVsUmF0aW87XG4gICAgdGhpcy5nYy5jbGVhclJlY3QoMCwgMCwgcGl4ZWxXaWR0aCwgcGl4ZWxIZWlnaHQpO1xuICB9XG4gIC8qKiBSZW5kZXIgYWxsIG9mIHRoZSBub24tb3ZlcmxhcHBpbmcgdmlzaWJsZSBsYWJlbHMgdG8gdGhlIGNhbnZhcy4gKi9cbiAgcHJpdmF0ZSBtYWtlTGFiZWxzKHJjOiBSZW5kZXJDb250ZXh0KSB7XG4gICAgaWYgKHJjLmxhYmVscyA9PSBudWxsIHx8IHJjLmxhYmVscy5wb2ludEluZGljZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLndvcmxkU3BhY2VQb2ludFBvc2l0aW9ucyA9PSBudWxsKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IGxyYyA9IHJjLmxhYmVscztcbiAgICBjb25zdCBzY2VuZUlzM0Q6IGJvb2xlYW4gPSByYy5jYW1lcmFUeXBlID09PSBDYW1lcmFUeXBlLlBlcnNwZWN0aXZlO1xuICAgIGNvbnN0IGxhYmVsSGVpZ2h0ID0gcGFyc2VJbnQodGhpcy5nYy5mb250LCAxMCk7XG4gICAgY29uc3QgZHByID0gd2luZG93LmRldmljZVBpeGVsUmF0aW87XG4gICAgbGV0IGdyaWQ6IENvbGxpc2lvbkdyaWQ7XG4gICAge1xuICAgICAgY29uc3QgcGl4dyA9IHRoaXMuY2FudmFzLndpZHRoICogZHByO1xuICAgICAgY29uc3QgcGl4aCA9IHRoaXMuY2FudmFzLmhlaWdodCAqIGRwcjtcbiAgICAgIGNvbnN0IGJiOiBCb3VuZGluZ0JveCA9IHtsb1g6IDAsIGhpWDogcGl4dywgbG9ZOiAwLCBoaVk6IHBpeGh9O1xuICAgICAgZ3JpZCA9IG5ldyBDb2xsaXNpb25HcmlkKGJiLCBwaXh3IC8gMjUsIHBpeGggLyA1MCk7XG4gICAgfVxuICAgIGxldCBvcGFjaXR5TWFwID0gZDNcbiAgICAgIC5zY2FsZVBvdygpXG4gICAgICAuZXhwb25lbnQoTWF0aC5FKVxuICAgICAgLmRvbWFpbihbcmMuZmFydGhlc3RDYW1lcmFTcGFjZVBvaW50WiwgcmMubmVhcmVzdENhbWVyYVNwYWNlUG9pbnRaXSlcbiAgICAgIC5yYW5nZShbMC4xLCAxXSk7XG4gICAgY29uc3QgY2FtUG9zID0gcmMuY2FtZXJhLnBvc2l0aW9uO1xuICAgIGNvbnN0IGNhbVRvVGFyZ2V0ID0gY2FtUG9zLmNsb25lKCkuc3ViKHJjLmNhbWVyYVRhcmdldCk7XG4gICAgbGV0IGNhbVRvUG9pbnQgPSBuZXcgVEhSRUUuVmVjdG9yMygpO1xuICAgIHRoaXMuZ2MudGV4dEJhc2VsaW5lID0gJ21pZGRsZSc7XG4gICAgdGhpcy5nYy5taXRlckxpbWl0ID0gMjtcbiAgICAvLyBIYXZlIGV4dHJhIHNwYWNlIGJldHdlZW4gbmVpZ2hib3JpbmcgbGFiZWxzLiBEb24ndCBwYWNrIHRvbyB0aWdodGx5LlxuICAgIGNvbnN0IGxhYmVsTWFyZ2luID0gMDtcbiAgICAvLyBTaGlmdCB0aGUgbGFiZWwgdG8gdGhlIHJpZ2h0IG9mIHRoZSBwb2ludCBjaXJjbGUuXG4gICAgY29uc3QgeFNoaWZ0ID0gNDtcbiAgICBjb25zdCBuID0gTWF0aC5taW4oTUFYX0xBQkVMU19PTl9TQ1JFRU4sIGxyYy5wb2ludEluZGljZXMubGVuZ3RoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG47ICsraSkge1xuICAgICAgbGV0IHBvaW50OiBUSFJFRS5WZWN0b3IzO1xuICAgICAge1xuICAgICAgICBjb25zdCBwaSA9IGxyYy5wb2ludEluZGljZXNbaV07XG4gICAgICAgIHBvaW50ID0gdXRpbC52ZWN0b3IzRnJvbVBhY2tlZEFycmF5KHRoaXMud29ybGRTcGFjZVBvaW50UG9zaXRpb25zLCBwaSk7XG4gICAgICB9XG4gICAgICAvLyBkaXNjYXJkIHBvaW50cyB0aGF0IGFyZSBiZWhpbmQgdGhlIGNhbWVyYVxuICAgICAgY2FtVG9Qb2ludC5jb3B5KGNhbVBvcykuc3ViKHBvaW50KTtcbiAgICAgIGlmIChjYW1Ub1RhcmdldC5kb3QoY2FtVG9Qb2ludCkgPCAwKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgbGV0IFt4LCB5XSA9IHV0aWwudmVjdG9yM0RUb1NjcmVlbkNvb3JkcyhcbiAgICAgICAgcmMuY2FtZXJhLFxuICAgICAgICByYy5zY3JlZW5XaWR0aCxcbiAgICAgICAgcmMuc2NyZWVuSGVpZ2h0LFxuICAgICAgICBwb2ludFxuICAgICAgKTtcbiAgICAgIHggKz0geFNoaWZ0O1xuICAgICAgLy8gQ29tcHV0aW5nIHRoZSB3aWR0aCBvZiB0aGUgZm9udCBpcyBleHBlbnNpdmUsXG4gICAgICAvLyBzbyB3ZSBhc3N1bWUgd2lkdGggb2YgMSBhdCBmaXJzdC4gVGhlbiwgaWYgdGhlIGxhYmVsIGRvZXNuJ3RcbiAgICAgIC8vIGNvbmZsaWN0IHdpdGggb3RoZXIgbGFiZWxzLCB3ZSBtZWFzdXJlIHRoZSBhY3R1YWwgd2lkdGguXG4gICAgICBjb25zdCB0ZXh0Qm91bmRpbmdCb3g6IEJvdW5kaW5nQm94ID0ge1xuICAgICAgICBsb1g6IHggLSBsYWJlbE1hcmdpbixcbiAgICAgICAgaGlYOiB4ICsgMSArIGxhYmVsTWFyZ2luLFxuICAgICAgICBsb1k6IHkgLSBsYWJlbEhlaWdodCAvIDIgLSBsYWJlbE1hcmdpbixcbiAgICAgICAgaGlZOiB5ICsgbGFiZWxIZWlnaHQgLyAyICsgbGFiZWxNYXJnaW4sXG4gICAgICB9O1xuICAgICAgaWYgKGdyaWQuaW5zZXJ0KHRleHRCb3VuZGluZ0JveCwgdHJ1ZSkpIHtcbiAgICAgICAgY29uc3QgdGV4dCA9IGxyYy5sYWJlbFN0cmluZ3NbaV07XG4gICAgICAgIGNvbnN0IGZvbnRTaXplID0gbHJjLmRlZmF1bHRGb250U2l6ZSAqIGxyYy5zY2FsZUZhY3RvcnNbaV0gKiBkcHI7XG4gICAgICAgIHRoaXMuZ2MuZm9udCA9IGZvbnRTaXplICsgJ3B4IHJvYm90byc7XG4gICAgICAgIC8vIE5vdywgY2hlY2sgd2l0aCBwcm9wZXJseSBjb21wdXRlZCB3aWR0aC5cbiAgICAgICAgdGV4dEJvdW5kaW5nQm94LmhpWCArPSB0aGlzLmdjLm1lYXN1cmVUZXh0KHRleHQpLndpZHRoIC0gMTtcbiAgICAgICAgaWYgKGdyaWQuaW5zZXJ0KHRleHRCb3VuZGluZ0JveCkpIHtcbiAgICAgICAgICBsZXQgb3BhY2l0eSA9IDE7XG4gICAgICAgICAgaWYgKHNjZW5lSXMzRCAmJiBscmMudXNlU2NlbmVPcGFjaXR5RmxhZ3NbaV0gPT09IDEpIHtcbiAgICAgICAgICAgIG9wYWNpdHkgPSBvcGFjaXR5TWFwKGNhbVRvUG9pbnQubGVuZ3RoKCkpO1xuICAgICAgICAgIH1cbiAgICAgICAgICB0aGlzLmdjLmZpbGxTdHlsZSA9IHRoaXMuc3R5bGVTdHJpbmdGcm9tUGFja2VkUmdiYShcbiAgICAgICAgICAgIGxyYy5maWxsQ29sb3JzLFxuICAgICAgICAgICAgaSxcbiAgICAgICAgICAgIG9wYWNpdHlcbiAgICAgICAgICApO1xuICAgICAgICAgIHRoaXMuZ2Muc3Ryb2tlU3R5bGUgPSB0aGlzLnN0eWxlU3RyaW5nRnJvbVBhY2tlZFJnYmEoXG4gICAgICAgICAgICBscmMuc3Ryb2tlQ29sb3JzLFxuICAgICAgICAgICAgaSxcbiAgICAgICAgICAgIG9wYWNpdHlcbiAgICAgICAgICApO1xuICAgICAgICAgIHRoaXMuZ2MubGluZVdpZHRoID0gTEFCRUxfU1RST0tFX1dJRFRIO1xuICAgICAgICAgIHRoaXMuZ2Muc3Ryb2tlVGV4dCh0ZXh0LCB4LCB5KTtcbiAgICAgICAgICB0aGlzLmdjLmxpbmVXaWR0aCA9IExBQkVMX0ZJTExfV0lEVEg7XG4gICAgICAgICAgdGhpcy5nYy5maWxsVGV4dCh0ZXh0LCB4LCB5KTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxuICBwcml2YXRlIHN0eWxlU3RyaW5nRnJvbVBhY2tlZFJnYmEoXG4gICAgcGFja2VkUmdiYUFycmF5OiBVaW50OEFycmF5LFxuICAgIGNvbG9ySW5kZXg6IG51bWJlcixcbiAgICBvcGFjaXR5OiBudW1iZXJcbiAgKTogc3RyaW5nIHtcbiAgICBjb25zdCBvZmZzZXQgPSBjb2xvckluZGV4ICogMztcbiAgICBjb25zdCByID0gcGFja2VkUmdiYUFycmF5W29mZnNldF07XG4gICAgY29uc3QgZyA9IHBhY2tlZFJnYmFBcnJheVtvZmZzZXQgKyAxXTtcbiAgICBjb25zdCBiID0gcGFja2VkUmdiYUFycmF5W29mZnNldCArIDJdO1xuICAgIHJldHVybiAncmdiYSgnICsgciArICcsJyArIGcgKyAnLCcgKyBiICsgJywnICsgb3BhY2l0eSArICcpJztcbiAgfVxuICBvblJlc2l6ZShuZXdXaWR0aDogbnVtYmVyLCBuZXdIZWlnaHQ6IG51bWJlcikge1xuICAgIGxldCBkcHIgPSB3aW5kb3cuZGV2aWNlUGl4ZWxSYXRpbztcbiAgICB0aGlzLmNhbnZhcy53aWR0aCA9IG5ld1dpZHRoICogZHByO1xuICAgIHRoaXMuY2FudmFzLmhlaWdodCA9IG5ld0hlaWdodCAqIGRwcjtcbiAgICB0aGlzLmNhbnZhcy5zdHlsZS53aWR0aCA9IG5ld1dpZHRoICsgJ3B4JztcbiAgICB0aGlzLmNhbnZhcy5zdHlsZS5oZWlnaHQgPSBuZXdIZWlnaHQgKyAncHgnO1xuICB9XG4gIGRpc3Bvc2UoKSB7XG4gICAgdGhpcy5yZW1vdmVBbGxMYWJlbHMoKTtcbiAgICB0aGlzLmNhbnZhcyA9IG51bGw7XG4gICAgdGhpcy5nYyA9IG51bGw7XG4gIH1cbiAgb25Qb2ludFBvc2l0aW9uc0NoYW5nZWQobmV3UG9zaXRpb25zOiBGbG9hdDMyQXJyYXkpIHtcbiAgICB0aGlzLndvcmxkU3BhY2VQb2ludFBvc2l0aW9ucyA9IG5ld1Bvc2l0aW9ucztcbiAgICB0aGlzLnJlbW92ZUFsbExhYmVscygpO1xuICB9XG4gIG9uUmVuZGVyKHJjOiBSZW5kZXJDb250ZXh0KSB7XG4gICAgaWYgKCF0aGlzLmxhYmVsc0FjdGl2ZSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnJlbW92ZUFsbExhYmVscygpO1xuICAgIHRoaXMubWFrZUxhYmVscyhyYyk7XG4gIH1cbiAgc2V0U2NlbmUoc2NlbmU6IFRIUkVFLlNjZW5lKSB7fVxuICBvblBpY2tpbmdSZW5kZXIocmVuZGVyQ29udGV4dDogUmVuZGVyQ29udGV4dCkge31cbn1cbiJdfQ==