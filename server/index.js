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
    rooms[code] = { host: socket.id, guest: null, rematchVotes: 0 };
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
    setTimeout(() => {
      io.to(code).emit('game_start');
    }, 500);
  });

  socket.on('game_event', ({ code, type, data }) => {
    const room = rooms[code];
    if (!room) return;

    if (type === 'rematch_request') {
      room.rematchVotes = (room.rematchVotes || 0) + 1;
      if (room.rematchVotes >= 2) {
        room.rematchVotes = 0;
        io.to(code).emit('game_event', { type: 'rematch_go' });
      } else {
        // tell the other person their opponent wants a rematch
        socket.to(code).emit('game_event', { type: 'rematch_request' });
      }
    } else {
      socket.to(code).emit('game_event', { type, data });
    }
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