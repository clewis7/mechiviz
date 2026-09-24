/*************************************************************************************************
  renderview-h264.js

  Browser client for branchpoint's H.264 rendercanvas backend. This is
  rendercanvas' renderview-client.js with the display path swapped: frames
  arrive as H.264 access units and are decoded with WebCodecs into a <canvas>,
  rather than arriving as JPEG and being handed to an <img>.

  Everything about interaction is inherited from BaseRenderView, which attaches
  its pointer, key and resize listeners to whichever element it is given.

  Every frame is an IDR carrying its own parameter sets, so the decoder can be
  rebuilt at any time and dropped frames cost nothing.

  *************************************************************************************************/

/* global BaseRenderView WebSocket VideoDecoder EncodedVideoChunk */

const wrapperElement = document.getElementById('canvas')
const statusElement = document.getElementById('status')
let view = null
let websocket = null
let isActive = null

updateStatus()
openWebsocketConnection()
window.openWebsocketConnection = openWebsocketConnection

class H264RenderView extends BaseRenderView {
  constructor (wrapperElement) {
    wrapperElement.classList.add('renderview-wrapper')

    // Create view element
    const viewElement = document.createElement('canvas')
    viewElement.style.imageRendering = 'pixelated' // if size does not match, use nearest-neighbor
    viewElement.style.touchAction = 'none' // prevent default pan/zoom behavior

    // Instantiate
    super(viewElement, wrapperElement)
    this.setThrottle(20) // 20ms -> max 50 move/wheel events per second

    this.frames = []
    this.updatePending = false
    this.context2d = viewElement.getContext('2d')
    this.decoder = null
    this.decoderKey = null
    this.timestamp = 0
  }

  ensureDecoder (codec, width, height) {
    const key = `${codec} ${width}x${height}`
    if (this.decoder !== null && this.decoderKey === key) { return }

    if (this.decoder !== null) {
      try { this.decoder.close() } catch (err) { /* already closed */ }
    }
    this.decoderKey = key
    this.decoder = new VideoDecoder({
      output: (frame) => this.drawFrame(frame),
      error: (err) => console.error('VideoDecoder error:', err)
    })
    // No description is given, which selects Annex B, the form the encoder
    // emits. The parameter sets ride along in every frame.
    this.decoder.configure({ codec, optimizeForLatency: true })
    console.log(`configured VideoDecoder for ${key}`)
  }

  drawFrame (frame) {
    const width = frame.displayWidth
    const height = frame.displayHeight
    if (this.viewElement.width !== width || this.viewElement.height !== height) {
      this.viewElement.width = width
      this.viewElement.height = height
    }
    this.context2d.drawImage(frame, 0, 0)
    frame.close()
  }

  requestAnimationFrame () {
    if (!this.updatePending) {
      this.updatePending = true
      window.requestAnimationFrame(this.animate.bind(this))
    }
  }

  animate () {
    this.updatePending = false
    if (this.frames.length === 0) { return }

    // Pick the oldest frame from the stack
    const frame = this.frames.shift()
    if (!frame.buffers || frame.buffers.length === 0) { return }

    this.ensureDecoder(frame.codec, frame.width, frame.height)

    // Timestamps only have to increase; the server paces the stream.
    this.timestamp += 1000000 / 60
    try {
      this.decoder.decode(new EncodedVideoChunk({
        type: 'key',
        timestamp: this.timestamp,
        data: new Uint8Array(frame.buffers[0])
      }))
    } catch (err) {
      console.error('decode failed, rebuilding decoder:', err)
      this.decoderKey = null
    }

    // Let the server know we processed the frame (even if it's not shown yet)
    this.sendResponse(frame)

    if (this.frames.length > 0) { this.requestAnimationFrame() }
  }

  sendResponse (frame) {
    const event = { type: '_framefeedback', index: frame.index, timestamp: frame.timestamp, localtime: Date.now() / 1000 }
    this.onEvent(event)
  }

  onEvent (event) {
    if (websocket !== null) {
      websocket.send(JSON.stringify(event))
    }
  }
}

function updateStatus () {
  if (statusElement === null) { return }

  let activeText = ''
  if (isActive !== null) {
    activeText = isActive ? ' (active)' : '(passive)'
  }

  if (typeof VideoDecoder === 'undefined') {
    statusElement.innerHTML = "<span style='color:#900'>?</span> No WebCodecs in this browser"
  } else if (websocket === null) {
    statusElement.innerHTML = "<span style='color:#900'>?</span> Disconnected <button onclick='openWebsocketConnection()'>reconnect</button>"
  } else {
    statusElement.innerHTML = `<span style='color:#090'>+</span> Connected ${activeText}`
  }
}

function openWebsocketConnection () {
  if (typeof VideoDecoder === 'undefined') {
    console.error('This backend needs WebCodecs (Chrome/Edge 94+, Safari 16.4+, Firefox 130+).')
    updateStatus()
    return
  }

  const ws = new WebSocket('ws://' + window.location.host + window.location.pathname)
  // Frames go straight into a decoder, so skip the Blob round trip
  ws.binaryType = 'arraybuffer'

  ws.onopen = (e) => {
    console.log('websocket opened')
    websocket = ws
    window.websocket = ws // allow manual closing to mimic lost connection
    if (view === null) {
      view = new H264RenderView(wrapperElement)
      console.log('created H264RenderView')
    }
    updateStatus()
  }
  ws.onerror = (e) => {
    console.log(`websocket error: ${e}`)
    websocket = null
    updateStatus()
  }

  let pendingMsg
  ws.onmessage = (e) => {
    let msg = null

    // A message with buffers arrives as json followed by that many binaries
    if (typeof e.data === 'string' || e.data instanceof String) {
      msg = JSON.parse(e.data)
      if (msg.nbuffers && msg.nbuffers > 0) {
        pendingMsg = msg
        pendingMsg.buffers = []
        msg = null
      } else {
        pendingMsg = null // discard unfinished pending message (if any)
      }
    } else {
      if (pendingMsg !== null) {
        pendingMsg.buffers.push(e.data)
        if (pendingMsg.buffers.length >= pendingMsg.nbuffers) {
          msg = pendingMsg
          pendingMsg = null
        }
      }
    }

    if (msg === null) { return }

    if (msg.type === 'framebufferdata') {
      view.frames.push(msg)
      view.requestAnimationFrame()
    } else if (msg.type === 'active') {
      isActive = msg.value
      updateStatus()
    } else if (msg.type === 'cursor') {
      view.setCursor(msg.value)
    } else if (msg.type === 'title') {
      view.setTitle(msg.value)
    } else if (msg.type === 'css_width') {
      view.setCssWidth(msg.value)
    } else if (msg.type === 'css_height') {
      view.setCssHeight(msg.value)
    }
  }

  ws.onclose = (e) => {
    console.log(`websocket closed: ${e.reason} (${e.code})`)
    websocket = null
    updateStatus()
  }
}
