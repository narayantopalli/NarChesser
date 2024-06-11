#include "planes.hpp"

// creates a set of board representation planes for a position
std::unique_ptr<Bitboard[]> planes::toPlane(const Board& board, Color color) {
    auto plane = std::make_unique<Bitboard[]>(14);
    const Board& position = board;
    Bitboard repetition1;
    Bitboard repetition2;
    if (position.isRepetition(1)) {
        repetition1 = ~repetition1;
        if (position.isRepetition(2)) {
            repetition2 = ~repetition2;
        }
    }
    int index = 0;
    bool reverseRanks = (color == Color::BLACK);
    auto addPieces = [&plane, &position, color, &index, reverseRanks](Color pieceColor) {
        std::array<PieceType, 6> pieceTypes = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN, PieceType::KING};
        for (const auto& pieceType : pieceTypes) {
            Bitboard pieces = position.pieces(pieceType, pieceColor);
            plane[index] = (reverseRanks ? pieces.reverseRanks() : pieces);
            index += 1;
        }
    };

    addPieces(color);
    addPieces(~color);
    plane[12] = repetition1;
    plane[13] = repetition2;
    return plane;
}

// creates planes with additional positional information
std::unique_ptr<Bitboard[]> planes::extraPlanes(const Board& board) {
    auto plane = std::make_unique<Bitboard[]>(6);
    const Board& position = board;
    Color color = position.sideToMove();
    Bitboard colorBoard;
    Bitboard p1K;
    Bitboard p1Q;
    Bitboard p2K;
    Bitboard p2Q;
    // Bitboard p1A;
    // Bitboard p2A;
    Bitboard enPassant;

    // for (int s = 0; s < 64; ++s) {
    //     if (position.isAttacked(static_cast<chess::Square>(s), color)) {
    //         p1A.set(s);
    //     }
    //     if (position.isAttacked(static_cast<chess::Square>(s), ~color)) {
    //         p2A.set(s);
    //     }
    // }

    // if (color == Color::BLACK) {
    //     colorBoard =~ colorBoard;
    //     p1A = p1A.reverseRanks();
    //     p2A = p2A.reverseRanks();
    // }

    const chess::Board::CastlingRights castlingRights = position.castlingRights();
    if (castlingRights.has(color, chess::Board::CastlingRights::Side::KING_SIDE)) {p1K =~ p1K;}
    if (castlingRights.has(color, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {p1Q =~ p1Q;}
    if (castlingRights.has(~color, chess::Board::CastlingRights::Side::KING_SIDE)) {p2K =~ p2K;}
    if (castlingRights.has(~color, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {p2Q =~ p2Q;}

    const auto ep = position.enpassantSq();
    if (chess::Square::is_valid(ep.rank(), ep.file())) {
        enPassant = Bitboard(position.enpassantSq().file());
    }

    plane[0] = colorBoard;
    plane[1] = p1K;
    plane[2] = p1Q;
    plane[3] = p2K;
    plane[4] = p2Q;
    // plane[5] = p1A;
    // plane[6] = p2A;
    plane[5] = enPassant;

    return plane;
}

std::unique_ptr<Bitboard[]> planes::emptyPlane() {
    auto plane = std::make_unique<Bitboard[]>(14);
    for (int i = 0; i < 14; ++i) {
        plane[i] = Bitboard(0);
    }
    return plane;
}
