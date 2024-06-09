#include "encoder.hpp"

// Function to encode the state of a chess board into an array of Bitboards
EncodedState::EncodedState(const Node* node, const std::vector<chess::Board>& traversed, const uint8_t history) : history(history) {

    totalPlanes = 14 * history + 6;
    encodedState = std::make_unique<Bitboard[]>(totalPlanes);
    auto index = 0;
    auto a = planes::toPlane(node->state, node->state.sideToMove());
    for (uint8_t i = 0; i < 14; ++i) {
        encodedState[index] = a[i];
        ++index;
    }
    a.reset();
    const int num_prevs = node->prev_list.size();
    const int num_traversed = traversed.size();
    for (int lookBack = 1; lookBack < history; ++lookBack) {
        const int lb_amount = num_prevs - lookBack;
        if (lb_amount < 0) {
            const auto trav_index = num_traversed + lb_amount;
            if (trav_index >= 0) {
                const auto state = traversed[trav_index];
                auto b = planes::toPlane(state, state.sideToMove());
                for (uint8_t i = 0; i < 14; ++i) {
                    encodedState[index] = b[i];
                    ++index;
                }
                b.reset();
            }
            else {
                auto b = planes::emptyPlane();
                for (uint8_t i = 0; i < 14; ++i) {
                    encodedState[index] = b[i];
                    ++index;
                }
                b.reset();
            }
        }
        else {
            auto b = planes::toPlane(node->prev_list[lb_amount]->state, node->prev_list[lb_amount]->state.sideToMove());
            for (uint8_t i = 0; i < 14; ++i) {
                encodedState[index] = b[i];
                ++index;
            }
            b.reset();
        }
    }
    auto c = planes::extraPlanes(node->state);
    for (uint8_t i = 0; i < 6; ++i) {
        encodedState[index] = c[i];
        ++index;
    }
    c.reset();
}
