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

  socket.on('create_room', ({ code, bestOf, maxPlayers }) => {
    rooms[code] = {
      players: [socket.id],
      maxPlayers: maxPlayers || 2,
      bestOf: bestOf || 5,
      rematchVotes: new Set()
    };
    socket.join(code);
    socket.data.roomCode = code;
    socket.emit('room_created', code);
    io.to(code).emit('room_update', { players: rooms[code].players, maxPlayers: rooms[code].maxPlayers });
  });

  socket.on('join_room', (code) => {
    const room = rooms[code];
    if (!room) return socket.emit('error', 'Room not found');
    if (room.players.length >= room.maxPlayers) return socket.emit('error', 'Room full');

    room.players.push(socket.id);
    socket.join(code);
    socket.data.roomCode = code;
    socket.emit('room_joined', code);
    io.to(code).emit('room_update', { players: room.players, maxPlayers: room.maxPlayers });

    if (room.players.length === room.maxPlayers) {
      setTimeout(() => {
        io.to(code).emit('game_start', {
          bestOf: room.bestOf,
          players: room.players,
          maxPlayers: room.maxPlayers
        });
      }, 500);
    }
  });

  socket.on('game_event', ({ code, type, data }) => {
    const room = rooms[code];
    if (!room) return;

    if (type === 'rematch_request') {
      room.rematchVotes.add(socket.id);
      if (room.rematchVotes.size >= room.players.length) {
        room.rematchVotes.clear();
        io.to(code).emit('game_event', { type: 'rematch_go', data: { bestOf: room.bestOf } });
      } else {
        socket.to(code).emit('game_event', {
          type: 'rematch_vote',
          data: { votes: room.rematchVotes.size, needed: room.players.length },
          fromId: socket.id
        });
      }
    } else if (data && data.targetId) {
      // Targeted signalling message (WebRTC offer/answer/ice)
      io.to(data.targetId).emit('game_event', { type, data, fromId: socket.id });
    } else {
      // Broadcast to everyone else in the room
      socket.to(code).emit('game_event', { type, data, fromId: socket.id });
    }
  });

  socket.on('disconnect', () => {
    const code = socket.data.roomCode;
    if (!code || !rooms[code]) return;
    const room = rooms[code];
    room.players = room.players.filter(id => id !== socket.id);
    if (room.players.length === 0) {
      delete rooms[code];
    } else {
      io.to(code).emit('opponent_left', { id: socket.id });
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));