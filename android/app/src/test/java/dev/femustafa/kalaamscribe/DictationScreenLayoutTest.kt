package dev.femustafa.kalaamscribe

import android.app.Application
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.test.core.app.ApplicationProvider
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/**
 * Regression guard for dictation-screen clipping: a long transcript used to grow
 * the (non-scrollable) transcript card until the Copy button slid below the fold on
 * small phones. The transcript body is now capped and scrolls inside the card, so
 * the card (and therefore Copy + history) always stays on-screen. Red on the old
 * unbounded-text layout, green on the capped one.
 */
abstract class DictationScreenLayoutBase {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun copyButtonStaysOnScreenWithALongTranscript() {
        val app: Application = ApplicationProvider.getApplicationContext()
        val kalaam = KalaamApp.from(app)
        // ~2,400 chars: enough to overflow any phone screen when rendered unwrapped.
        kalaam.setTranscript("word ".repeat(400).trim())
        kalaam.setLastTranscriptionMs(1200L)

        val vm = DictationViewModel(app)
        composeRule.setContent {
            DictationScreen(viewModel = vm, onOpenSettings = {})
        }

        composeRule.onNodeWithText("Copy").assertIsDisplayed()
    }
}

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = "w411dp-h891dp-normal-long-notround")
class DictationScreenLargePhoneTest : DictationScreenLayoutBase()

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = "w360dp-h640dp-normal-notlong-notround")
class DictationScreenSmallPhoneTest : DictationScreenLayoutBase()

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = "w320dp-h480dp-normal-notlong-notround")
class DictationScreenTinyPhoneTest : DictationScreenLayoutBase()