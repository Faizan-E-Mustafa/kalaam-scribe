package dev.femustafa.kalaamscribe

import android.app.Application
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
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

    @Test
    fun fullTextShowsBeginningOfALongTranscript() {
        val app: Application = ApplicationProvider.getApplicationContext()
        val kalaam = KalaamApp.from(app)
        // ~2,400 chars: the card auto-scrolls to the end, so only the Full text dialog
        // can prove the beginning survived.
        kalaam.setTranscript("word ".repeat(400).trim())
        kalaam.setLastTranscriptionMs(1200L)

        val vm = DictationViewModel(app)
        composeRule.setContent {
            DictationScreen(viewModel = vm, onOpenSettings = {})
        }

        composeRule.onNodeWithText("Full text").performClick()
        composeRule.onNodeWithText("Transcript (1999 chars)").assertIsDisplayed()
        // The dialog starts scrolled at the top, so the first words are in view.
        composeRule.onNodeWithTag("fullTranscriptText")
            .assert(hasText("word word word word word word word word", substring = true))
    }

    @Test
    fun scrollbarShownWhenTranscriptIsLong() {
        val app: Application = ApplicationProvider.getApplicationContext()
        val kalaam = KalaamApp.from(app)
        kalaam.setTranscript("word ".repeat(400).trim())
        kalaam.setLastTranscriptionMs(1200L)

        val vm = DictationViewModel(app)
        composeRule.setContent {
            DictationScreen(viewModel = vm, onOpenSettings = {})
        }

        // Long text overflows the capped card, so the visible scrollbar must be there —
        // it is the explicit "the text is long" affordance for the user.
        composeRule.onAllNodesWithTag("transcriptScrollbar").assertCountEquals(1)
    }

    @Test
    fun scrollbarHiddenWhenTranscriptIsShort() {
        val app: Application = ApplicationProvider.getApplicationContext()
        val kalaam = KalaamApp.from(app)
        kalaam.setTranscript("just a short transcript")
        kalaam.setLastTranscriptionMs(1200L)

        val vm = DictationViewModel(app)
        composeRule.setContent {
            DictationScreen(viewModel = vm, onOpenSettings = {})
        }

        composeRule.onAllNodesWithTag("transcriptScrollbar").assertCountEquals(0)
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