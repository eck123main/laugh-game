const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

app.use(express.static(path.join(__dirname, '../client')));

const rooms = {};

io.on('connection', (socket) => {
  console.log('connected:', socket.id);

  socket.on('create_room', (code) => {
    rooms[code] = { host: socket.id, guest: null };
    socket.join(code);
    socket.emit('room_created', code);
  });

socket.on('join_room', (code) => {
  const room = rooms[code];
  if (!room) return socket.emit('error', 'Room not found');
  if (room.guest) return socket.emit('error', 'Room full');
  room.guest = socket.id;
  socket.join(code);
  socket.emit('room_joined', code);
  // Small delay to ensure both sides are ready
  setTimeout(() => {
    io.to(code).emit('game_start');
  }, 500);
});

  socket.on('game_event', ({ code, type, data }) => {
  socket.to(code).emit('game_event', { type, data }); // data forwarded ✓
});

  socket.on('disconnect', () => {
    for (const code in rooms) {
      const room = rooms[code];
      if (room.host === socket.id || room.guest === socket.id) {
        io.to(code).emit('opponent_left');
        delete rooms[code];
      }
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));