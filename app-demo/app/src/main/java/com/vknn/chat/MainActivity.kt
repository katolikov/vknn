package com.vknn.chat

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.vknn.chat.ui.ChatScreen
import com.vknn.chat.ui.VknnChatTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VknnChatTheme {
                val vm: ChatViewModel = viewModel()
                val ui by vm.ui.collectAsState()
                ChatScreen(
                    ui = ui,
                    onDownload = vm::download,
                    onLoad = vm::loadModel,
                    onSend = vm::send,
                    onReset = vm::reset,
                    onTemp = vm::setTemperature,
                )
            }
        }
    }
}
