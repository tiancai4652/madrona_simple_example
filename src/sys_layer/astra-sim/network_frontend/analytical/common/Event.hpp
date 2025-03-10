/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "Type.hpp"
#include "../../../../../sys_layer/containers/FixedTuple.hpp"
#include "../../../../../sys_layer/containers/Pair.hpp"


namespace NetworkAnalytical {

/**
 * Event is a wrapper for a callback function and its argument.
 */
class Event {
  public:
    /**
     * Constructor.
     *
     * @param callback function pointer
     * @param callback_arg argument of the callback function
     */
    Event(Callback callback, CallbackArg callback_arg) noexcept;

    /**
     * Invoke the callback function.
     */
    void invoke_event() noexcept;

    /**
     * Get the callback function and the argument.
     *
     * @return callback function and its argument
     */
    [[nodiscard]] custom::Pair<Callback, CallbackArg> get_handler_arg() const noexcept;

  private:
    /// pointer to the callback function
    Callback callback;

    /// argument of the callback function
    CallbackArg callback_arg;
};

}  // namespace NetworkAnalytical
