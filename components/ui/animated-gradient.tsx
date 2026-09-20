"use client"

import { useEffect, useRef } from "react"

type AnimatedGradientProps = {
  className?: string
  colors?: [string, string, string]
  speed?: number
}

const vertex = `attribute vec2 position; void main(){ gl_Position=vec4(position,0.0,1.0); }`
const fragment = `precision mediump float;
uniform vec2 resolution; uniform float time; uniform vec3 color1; uniform vec3 color2; uniform vec3 color3;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
void main(){vec2 uv=gl_FragCoord.xy/resolution;vec2 p=(gl_FragCoord.xy-.5*resolution)/min(resolution.x,resolution.y);float t=time*.12;float n=noise(p*2.0+vec2(t,-t*.7));float bands=.5+.5*sin(p.x*2.1+p.y*1.3+t+n*3.0);vec3 c=mix(color1,color2,smoothstep(.12,.62,bands));c=mix(c,color3,smoothstep(.56,1.,bands)*.55);float glow=1.-smoothstep(.25,1.1,length(uv-.5));gl_FragColor=vec4(c*(.72+glow*.28),1.);}`

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "")
  const normalized = value.length === 3 ? value.split("").map((part) => part + part).join("") : value
  return [0, 2, 4].map((index) => parseInt(normalized.slice(index, index + 2), 16) / 255) as [number, number, number]
}

export default function AnimatedGradient({ className, colors = ["#06131f", "#0b4d61", "#65d8d0"], speed = 1 }: AnimatedGradientProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const gl = canvas?.getContext("webgl")
    if (!canvas || !gl) return
    const compile = (type: number, source: string) => { const shader = gl.createShader(type); if (!shader) return null; gl.shaderSource(shader, source); gl.compileShader(shader); return shader }
    const vs = compile(gl.VERTEX_SHADER, vertex)
    const fs = compile(gl.FRAGMENT_SHADER, fragment)
    if (!vs || !fs) return
    const program = gl.createProgram(); if (!program) return
    gl.attachShader(program, vs); gl.attachShader(program, fs); gl.linkProgram(program); gl.useProgram(program)
    const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW)
    const position = gl.getAttribLocation(program, "position"); gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
    const resolution = gl.getUniformLocation(program, "resolution"); const time = gl.getUniformLocation(program, "time")
    const colorLocations = ["color1", "color2", "color3"].map((name) => gl.getUniformLocation(program, name))
    const rgb = colors.map(hexToRgb)
    const resize = () => { const dpr = Math.min(window.devicePixelRatio || 1, 1.5); canvas.width = Math.max(1, canvas.clientWidth * dpr); canvas.height = Math.max(1, canvas.clientHeight * dpr); gl.viewport(0, 0, canvas.width, canvas.height) }
    const started = performance.now(); let frame = 0
    const render = (now: number) => { gl.uniform2f(resolution, canvas.width, canvas.height); gl.uniform1f(time, (now - started) / 1000 * speed); rgb.forEach((color, index) => gl.uniform3fv(colorLocations[index], color)); gl.drawArrays(gl.TRIANGLES, 0, 3); frame = requestAnimationFrame(render) }
    resize(); window.addEventListener("resize", resize); frame = requestAnimationFrame(render)
    return () => { cancelAnimationFrame(frame); window.removeEventListener("resize", resize); gl.deleteBuffer(buffer); gl.deleteProgram(program) }
  }, [colors, speed])

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />
}
